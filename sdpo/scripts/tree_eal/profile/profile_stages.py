"""Per-stage profile of the 2cc8cfa Tree-EAL training step (eal_mode=dense_tree,
--dense_eal_start --exact_starts --top4_start_eal [--prob_starts], path_topk 4),
fwd + bwd, no optimizer.

Stages (synchronized wall clock):
  tgt_prefix   target prefix forward over the whole sequence (no grad)
  drf_prefix   draft prefix forward (grad)
  exact        _exact_window_starts: draft tree expansion at supervised positions (no grad)
  dense_loss   dense_tree_loss: teacher-forced dense KL + top-4 E[L] at the starts
  rollout      _rollout_batch (draft_steps + verify + rollout KL), anchor tree
  bwd_head     backward until the draft prefix hidden gets its grad (dense loss + rollout)
  bwd_prefix   rest of the backward (draft prefix)
"""
import argparse, json, os, sys, time, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--prob_starts", action="store_true")
ap.add_argument("--segment_frac", type=float, default=0.0)
ap.add_argument("--random_frac", type=float, default=0.0)
ap.add_argument("--micro_bs", type=int, default=1)
ap.add_argument("--rollout", default="grad")
ap.add_argument("--forest", action="store_true")
ap.add_argument("--n_steps", type=int, default=6)
ap.add_argument("--data", required=True, help="ShareGPT-format jsonl (e.g. gen_alpaca_regen.py output)")
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--trace", action="store_true")
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import torch
from transformers import AutoTokenizer
import small_lm_tree_v2_model as M
from data_pipeline import build_dataset

m = M.SmallLMTreeV2Model("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", [4, 3, 2, 1, 1, 1, 1],
                         dtype=torch.bfloat16, tree_budget=None).cuda()
m.dense_eal_start = True
m.exact_starts = True
m.exact_starts_chunk = args.chunk
m.top4_start_eal = True
m.prob_starts = args.prob_starts
m.exact_segment_frac = args.segment_frac
m.exact_burn = 32
m.exact_random_frac = args.random_frac
m.forest = args.forest
m.rollout = args.rollout
m.train()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
ds = build_dataset(tok, args.data, 2048, max_samples=args.n_steps * args.micro_bs)

T = defaultdict(float)
st = {"on": False, "inner": 0, "hook_t": None}


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def wrap(name, fn, top_only=False):
    def w(*a, **k):
        if not st["on"] or (top_only and st["inner"]):
            return fn(*a, **k)
        st["inner"] += 1
        t = sync()
        try:
            return fn(*a, **k)
        finally:
            T[name] += sync() - t
            st["inner"] -= 1
    return w


m.target_model.model.forward = wrap("tgt_prefix", m.target_model.model.forward, top_only=True)
_dmf = m.draft_model.model.forward


def dmf(*a, **k):
    if not st["on"] or st["inner"]:
        return _dmf(*a, **k)
    t = sync(); out = _dmf(*a, **k); T["drf_prefix"] += sync() - t
    h = out.last_hidden_state
    if h.requires_grad:
        h.register_hook(lambda g: st.__setitem__("hook_t", sync()) or None)
    return out


m.draft_model.model.forward = dmf
M.SmallLMTreeV2Model._exact_window_starts = wrap("exact", M.SmallLMTreeV2Model._exact_window_starts)
M.SmallLMTreeV2Model._rollout_batch = wrap("rollout", M.SmallLMTreeV2Model._rollout_batch)
M.dense_tree_loss = wrap("dense_loss", M.dense_tree_loss)


def batch(i):
    idx = range(i * args.micro_bs, (i + 1) * args.micro_bs)
    L = max(len(ds[j]["input_ids"]) for j in idx)
    ids = torch.zeros(args.micro_bs, L, dtype=torch.long)
    am = torch.zeros_like(ids); lm = torch.zeros_like(ids)
    for r, j in enumerate(idx):
        n = len(ds[j]["input_ids"])
        ids[r, :n] = torch.tensor(ds[j]["input_ids"]); am[r, :n] = 1
        lm[r, :n] = torch.tensor(ds[j]["loss_mask"])
    return ids.cuda(), am.cuda(), lm.cuda(), int(am.sum())


def step(i):
    torch.manual_seed(i)
    ids, am, lm, ntok = batch(i)
    t0 = sync()
    loss, _, met = m(ids, am, lm, anchor_kl_coef=1.0, eal_aux_coef=0.1, temperature=1.0,
                     eal_mode="dense_tree", path_topk=4, n_anchors=1)
    t1 = sync()
    st["hook_t"] = None
    loss.backward()
    t2 = sync()
    m.draft_model.zero_grad(set_to_none=True)
    return t0, t1, t2, ntok


for p in range(2):                      # pass 0 warmup
    st["on"] = p == 1
    T.clear()
    torch.cuda.reset_peak_memory_stats()
    toks = 0
    for i in range(args.n_steps):
        t0, t1, t2, ntok = step(i)
        if st["on"]:
            toks += ntok
            T["fwd"] += t1 - t0
            T["bwd"] += t2 - t1
            if st["hook_t"] is not None:
                T["bwd_head"] += st["hook_t"] - t1
                T["bwd_prefix"] += t2 - st["hook_t"]
n = args.n_steps
ms = {k: 1000 * v / n for k, v in T.items()}
ms["fwd_other"] = ms["fwd"] - sum(ms.get(k, 0) for k in
                                  ("tgt_prefix", "drf_prefix", "exact", "dense_loss", "rollout"))
ms["step"] = ms["fwd"] + ms["bwd"]
res = {"cfg": vars(args), "ms": ms, "tokens_per_step": toks / n,
       "peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
       "exact_rounds": getattr(m, "_exact_rounds", None)}
print(f"== {args.out} prob_starts={args.prob_starts} seg={args.segment_frac} rnd={args.random_frac} "
      f"micro_bs={args.micro_bs} rollout={args.rollout} tokens/step={toks / n:.0f} "
      f"peak={res['peak_GiB']:.1f}GiB")
for k in ("tgt_prefix", "drf_prefix", "exact", "dense_loss", "rollout", "fwd_other",
          "bwd_head", "bwd_prefix", "step"):
    v = ms.get(k, 0.0)
    print(f"  {k:11s} {v:8.1f} ms  ({v / ms['step']:5.1%})")

if args.trace:
    from torch.profiler import profile, ProfilerActivity
    st["on"] = False
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = sync()
        for i in range(2):
            step(i)
        wall = sync() - t0
    evs = [e for e in prof.events() if e.device_type.name == "CUDA"]
    busy, end = 0.0, 0.0
    for s, e in sorted((e.time_range.start, e.time_range.end) for e in evs):
        if e <= end:
            continue
        busy += e - max(s, end); end = e
    by = defaultdict(lambda: [0.0, 0])
    for e in evs:
        by[e.name][0] += e.time_range.elapsed_us(); by[e.name][1] += 1
    res["trace"] = {"wall_ms_per_step": wall * 500, "gpu_busy_ms_per_step": busy / 2000,
                    "kernels_per_step": len(evs) / 2}
    print(f"  trace: wall {wall * 500:.0f} ms/step, GPU busy {busy / 2000:.0f} ms/step "
          f"({busy / 1000 / (wall * 1000):.0%}), {len(evs) / 2:.0f} kernels/step")
    for name, (us, c) in sorted(by.items(), key=lambda x: -x[1][0])[:12]:
        print(f"    {us / 2000:7.1f} ms  x{c // 2:5d}  {name[:90]}")
json.dump(res, open(args.out, "w"), indent=1)
