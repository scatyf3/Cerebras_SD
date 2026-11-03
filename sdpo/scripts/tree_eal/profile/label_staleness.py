"""How stale would cached --exact_starts labels get? Runs the no-grad stage (all positions expanded)
with the base draft and with a trained draft, same target and sequences, and compares
n(t), the per-path accept lengths and the window starts.

Base Qwen3-0.6B vs the draft after 219 steps of sample_minpq (lr 1e-6, ~2 epochs), 12 val
sequences: n(t) identical 92.5%, all 4 path_n identical 65.2%, starts Jaccard 85.2%.
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="trained draft (HF directory)")
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--chunk", type=int, default=32)
ap.add_argument("--path_topk", type=int, default=4)
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import small_lm_tree_v2_model as M
from data_pipeline import build_dataset

m = M.SmallLMTreeV2Model("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", [4, 3, 2, 1, 1, 1, 1],
                         dtype=torch.bfloat16).cuda().eval()
base = {k: v.clone() for k, v in m.draft_model.state_dict().items()}
trained = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16).state_dict()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
ds = build_dataset(tok, args.data, 2048, max_samples=args.n)


def stage1(ids, lm):
    with torch.no_grad():
        th = m.target_model.model(input_ids=ids, return_dict=True).last_hidden_state
        do = m.draft_model.model(input_ids=ids, use_cache=True, return_dict=True)
        k, v = M._extract_kv_lists(do.past_key_values)
        st, n = m._exact_window_starts(do.last_hidden_state, k, v, th, lm, args.path_topk,
                                       chunk=args.chunk, all_positions=True)
    return st[0].cpu(), n[0].cpu(), m._exact_path_n[0].cpu()


agg = dict(pos=0, n_same=0, pn_same=0, st_a=0, st_b=0, st_both=0, nA=0.0, nB=0.0)
for i in range(len(ds)):
    ids = torch.tensor([ds[i]["input_ids"]]).cuda()
    lm = torch.tensor([ds[i]["loss_mask"]]).cuda()
    m.draft_model.load_state_dict(base)
    a = stage1(ids, lm)
    m.draft_model.load_state_dict(trained, strict=False)
    b = stage1(ids, lm)
    mask = lm[0, 1:].bool().cpu()
    agg["pos"] += int(mask.sum())
    agg["n_same"] += int(((a[1] == b[1]) & mask).sum())
    agg["pn_same"] += int(((a[2] == b[2]).all(-1) & mask).sum())
    agg["st_a"] += int(a[0].sum()); agg["st_b"] += int(b[0].sum())
    agg["st_both"] += int((a[0] & b[0]).sum())
    agg["nA"] += float(a[1][a[0]].float().sum()); agg["nB"] += float(b[1][b[0]].float().sum())
p = agg["pos"]
print(f"{len(ds)} sequences, {p} supervised positions")
print(f"n(t) identical: {agg['n_same'] / p:.1%}   all path_n identical: {agg['pn_same'] / p:.1%}")
print(f"starts: base {agg['st_a']}, trained {agg['st_b']}, shared {agg['st_both']} "
      f"(Jaccard {agg['st_both'] / (agg['st_a'] + agg['st_b'] - agg['st_both']):.1%})")
print(f"mean accept at starts: base {agg['nA'] / agg['st_a']:.3f}, trained {agg['nB'] / agg['st_b']:.3f}")
