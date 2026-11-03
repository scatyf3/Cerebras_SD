"""NVTX-annotated Tree-EAL training step (2cc8cfa: dense_tree + exact_starts + top4_start_eal),
for nsys with --gpu-metrics-devices. Stage boundaries are synchronized so each NVTX range is
also the GPU window of its stage. Only the measured steps are inside cudaProfilerStart/Stop.

Ranges (nested):
  step
    tgt_prefix          target prefix forward (no grad)
    drf_prefix          draft prefix forward (grad)
    exact               _exact_window_starts (no grad)
      exact/approx      _top4path_accept on teacher-forced draft logits
      exact/expand rK   one _expand_positions call (round K)
    dense_loss          dense_tree_loss (teacher-forced KL + top-4 E[L])
    rollout             _rollout_batch (anchor tree)
      rollout/draft     draft tree forwards
      rollout/verify    target tree forward
    bwd_head            backward until the draft prefix hidden has its grad
    bwd_prefix          rest of the backward
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--prob_starts", action="store_true")
ap.add_argument("--n_warm", type=int, default=1)
ap.add_argument("--n_steps", type=int, default=2)
ap.add_argument("--first", type=int, default=0, help="first dataset index to use")
ap.add_argument("--data", required=True)
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import torch
from torch.cuda import nvtx
from transformers import AutoTokenizer
import small_lm_tree_v2_model as M
from data_pipeline import build_dataset

m = M.SmallLMTreeV2Model("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", [4, 3, 2, 1, 1, 1, 1],
                         dtype=torch.bfloat16, tree_budget=None).cuda()
m.dense_eal_start = m.exact_starts = m.top4_start_eal = True
m.exact_starts_chunk = 64
m.prob_starts = args.prob_starts
m.exact_segment_frac = 0.0
m.exact_burn = 32
m.exact_random_frac = 0.0
m.train()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
n_all = args.n_warm + args.n_steps
ds = build_dataset(tok, args.data, 2048, max_samples=args.first + n_all)

st = {"depth": 0, "round": 0, "in_rollout": False}


def rng(name, fn, top_only=False):
    def w(*a, **k):
        if top_only and st["depth"]:
            return fn(*a, **k)
        torch.cuda.synchronize()
        rid = nvtx.range_start(name if name != "expand" else f"exact/expand r{st['round']} n={len(a[1])}")
        st["depth"] += 1
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize()
            st["depth"] -= 1
            nvtx.range_end(rid)
    return w


m.target_model.model.forward = rng("tgt_prefix", m.target_model.model.forward, top_only=True)
m.draft_model.model.forward = rng("drf_prefix", m.draft_model.model.forward, top_only=True)
_ex = M.SmallLMTreeV2Model._exact_window_starts


def exact(self, *a, **k):
    st["round"] = 0
    return rng("exact", _ex)(self, *a, **k)


M.SmallLMTreeV2Model._exact_window_starts = exact
_ep = M.SmallLMTreeV2Model._expand_positions


def expand(self, rows, cols, *a, **k):
    st["round"] += 1
    return rng("expand", lambda s, r, c, *aa, **kk: _ep(s, r, c, *aa, **kk))(self, rows, cols, *a, **k)


M.SmallLMTreeV2Model._expand_positions = expand
M._top4path_accept = rng("exact/approx", M._top4path_accept)
M.dense_tree_loss = rng("dense_loss", M.dense_tree_loss)
_rb = M.SmallLMTreeV2Model._rollout_batch


def rollout(self, *a, **k):
    st["in_rollout"] = True
    try:
        return rng("rollout", _rb)(self, *a, **k)
    finally:
        st["in_rollout"] = False


M.SmallLMTreeV2Model._rollout_batch = rollout
_dfwd, _tfwd = m.draft_model.forward, m.target_model.forward
m.draft_model.forward = lambda *a, **k: (rng("rollout/draft", _dfwd) if st["in_rollout"] else _dfwd)(*a, **k)
m.target_model.forward = lambda *a, **k: (rng("rollout/verify", _tfwd) if st["in_rollout"] else _tfwd)(*a, **k)

# bwd split: the draft prefix hidden's grad hook ends bwd_head
_dmf = m.draft_model.model.forward


def dmf(*a, **k):
    out = _dmf(*a, **k)
    h = out.last_hidden_state
    if h.requires_grad and not st["depth"]:
        def hook(g):
            torch.cuda.synchronize()
            nvtx.range_end(st["bwd_id"])
            st["bwd_id"] = nvtx.range_start("bwd_prefix")
        h.register_hook(hook)
    return out


m.draft_model.model.forward = dmf


def batch(i):
    ex = ds[args.first + i]
    ids = torch.tensor([ex["input_ids"]], device="cuda")
    return ids, torch.ones_like(ids), torch.tensor([ex["loss_mask"]], device="cuda")


def step(i):
    torch.manual_seed(i)
    ids, am, lm = batch(i)
    torch.cuda.synchronize()
    sid = nvtx.range_start(f"step {i} tokens={ids.size(1)}")
    loss, _, _ = m(ids, am, lm, anchor_kl_coef=1.0, eal_aux_coef=0.1, temperature=1.0,
                   eal_mode="dense_tree", path_topk=4, n_anchors=1)
    torch.cuda.synchronize()
    st["bwd_id"] = nvtx.range_start("bwd_head")
    loss.backward()
    torch.cuda.synchronize()
    nvtx.range_end(st["bwd_id"])
    nvtx.range_end(sid)
    m.draft_model.zero_grad(set_to_none=True)


for i in range(args.n_warm):
    step(i)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
t0 = time.perf_counter()
for i in range(args.n_warm, n_all):
    step(i)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"measured {args.n_steps} steps in {time.perf_counter() - t0:.2f} s")
