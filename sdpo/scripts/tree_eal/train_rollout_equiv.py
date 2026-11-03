"""Single-GPU Tree-EAL draft training, mirroring sdpo/main_small_lm_tree_v2.py with
sdpo/scripts/tree_eal/ds_zero2_a6000_3gpu.json (no DeepSpeed):
  AdamW(betas 0.9/0.95, wd 0.01) on fp32 master weights of the bf16 draft,
  DeepSpeed WarmupDecayLR (log warmup over 10% of steps, linear decay to 0),
  grad clip 1.0, micro_bs 1, grad_accum 6, loss / grad_accum.
Checks that --rollout replay/graph train the same draft as the original
--rollout grad: only --rollout differs between runs. Every micro step reseeds torch with
(seed, step), so runs with the same seed draw the same anchors / samples.

Eval (each epoch, held-out set, no_grad, original rollout code for every run):
  exact E[accepted length | tree] under the ratio verify rule (the eval rule),
  for n_anchors=4 trees per sample sampled with a fixed per-sample seed.
"""
import argparse, json, math, os, sys, time, warnings
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--rollout", default="grad")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--train", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--grad_accum", type=int, default=6)
ap.add_argument("--eal_mode", default="sample_minpq")
ap.add_argument("--max_len", type=int, default=2048)
ap.add_argument("--eval_only", action="store_true")
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import torch
from transformers import AutoTokenizer
from small_lm_tree_v2_model import SmallLMTreeV2Model
from data_pipeline import build_dataset

os.makedirs(args.out, exist_ok=True)
log = open(os.path.join(args.out, "log.jsonl"), "w")
m = SmallLMTreeV2Model("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", [4, 3, 2, 1, 1, 1, 1],
                       dtype=torch.bfloat16, tree_budget=128).cuda()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
train = build_dataset(tok, args.train, args.max_len)
val = build_dataset(tok, args.val, args.max_len)
print(f"train {len(train)} val {len(val)}", flush=True)


def tensors(ex):
    ids = torch.tensor([ex["input_ids"]], device="cuda")
    return ids, torch.ones_like(ids), torch.tensor([ex["loss_mask"]], device="cuda")


@torch.no_grad()
def evaluate(tag):
    m.eval()
    m.rollout, m.forest = "grad", False
    el, draw = [], []
    for i in range(len(val)):
        torch.manual_seed(777 + i)
        ids, am, lm = tensors(val[i])
        # sample_anyk with one draw: eal_loss = -E[L | tree] under independent
        # min(1, p/q) acceptance, the ratio verify rule (score term is 0).
        _, _, met = m(ids, am, lm, anchor_kl_coef=0.0, eal_aux_coef=0.0,
                      eal_mode="sample_anyk", n_anchors=4)
        if met["num_anchors"]:
            el.append(-met["eal_loss"]); draw.append(met["eal_mean"])
    r = {"tag": tag, "val_exact_EL": sum(el) / len(el), "val_draw_AL": sum(draw) / len(draw),
         "n": len(el)}
    print(json.dumps(r), flush=True)
    log.write(json.dumps(r) + "\n"); log.flush()
    m.train()
    return r


if args.eval_only:
    evaluate("base")
    sys.exit()

params = [p for p in m.draft_model.parameters() if p.requires_grad]
master = [p.detach().float().clone().requires_grad_(True) for p in params]
opt = torch.optim.AdamW(master, lr=0.0, betas=(0.9, 0.95), weight_decay=0.01)
total = args.epochs * len(train) // args.grad_accum
warm = max(total // 10, 1)


def lr_at(it):          # DeepSpeed WarmupDecayLR, warmup_type="log" (default)
    if it < warm:
        return args.lr * math.log(it + 1) / math.log(warm)
    return args.lr * max(0.0, (total - it) / max(1.0, total - warm))


evaluate("step0")
m.rollout = args.rollout
m.train()
step, micro, t_run = 0, 0, time.time()
acc = {"eal_mean": 0.0, "eal_loss": 0.0, "n": 0, "t": 0.0}
for ep in range(args.epochs):
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(ep)).tolist()
    for idx in order:
        m.rollout = args.rollout
        torch.manual_seed(args.seed * 1000003 + micro)
        ids, am, lm = tensors(train[idx])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss, _, met = m(ids, am, lm, anchor_kl_coef=0.0, eal_aux_coef=0.0,
                         temperature=1.0, eal_mode=args.eal_mode, n_anchors=1)
        (loss / args.grad_accum).backward()
        for p, mp in zip(params, master):
            if p.grad is not None:
                g = p.grad.float()
                mp.grad = g if mp.grad is None else mp.grad + g
                p.grad = None
        torch.cuda.synchronize()
        acc["t"] += time.perf_counter() - t0
        acc["eal_mean"] += met["eal_mean"]; acc["eal_loss"] += met["eal_loss"]; acc["n"] += 1
        micro += 1
        if micro % args.grad_accum == 0:
            gn = torch.nn.utils.clip_grad_norm_(master, 1.0).item()
            lr = lr_at(step)
            for g_ in opt.param_groups:
                g_["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for p, mp in zip(params, master):
                    p.copy_(mp)                 # in place: graph rollouts read by address
            step += 1
            if step % 5 == 0:
                n = acc["n"]
                r = {"step": step, "epoch": ep, "lr": lr, "grad_norm": gn,
                     "train_EL_minpq": -acc["eal_loss"] / n, "train_draw_AL": acc["eal_mean"] / n,
                     "fwd_bwd_ms": 1000 * acc["t"] / n}
                log.write(json.dumps(r) + "\n"); log.flush()
                print(json.dumps(r), flush=True)
                acc = {"eal_mean": 0.0, "eal_loss": 0.0, "n": 0, "t": 0.0}
    evaluate(f"epoch{ep}")
m.draft_model.save_pretrained(os.path.join(args.out, "hf"))
tok.save_pretrained(os.path.join(args.out, "hf"))
log.write(json.dumps({"done": True, "wall_s": time.time() - t_run}) + "\n")
print("done", time.time() - t_run)
