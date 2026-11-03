"""--exact_starts takes the target's greedy token at depth j of a window at s to be the argmax of
its prefix logits at s + j, i.e. it assumes the accepted tokens are the sequence's own tokens.
That holds only when the data is the target's greedy output. This measures, on the given data:
  - how often the target argmax differs from the next data token, and
  - how many windows (all positions expanded) are judged in a context that leaves the data:
    'path' if an accepted token differs from the data, 'decisive' if the check that ends the
    window does too.
On a temperature-1.0 Qwen3-8B Alpaca regen sequence: 14.8% of positions differ (6 sequences),
and 16 of 74 windows (22%) are decisive-affected.
"""
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--path_topk", type=int, default=4)
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import torch
from transformers import AutoTokenizer
import small_lm_tree_v2_model as M
from data_pipeline import build_dataset

m = M.SmallLMTreeV2Model("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B", [4, 3, 2, 1, 1, 1, 1],
                         dtype=torch.bfloat16).cuda().eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
ds = build_dataset(tok, args.data, 2048, max_samples=args.n)
mis_tot = sup_tot = win_tot = path_tot = dec_tot = 0
for i in range(len(ds)):
    ids = torch.tensor([ds[i]["input_ids"]]).cuda()
    lm = torch.tensor([ds[i]["loss_mask"]]).cuda()
    with torch.no_grad():
        th = m.target_model.model(input_ids=ids, return_dict=True).last_hidden_state
        u = m.target_model.lm_head(th[0, :-1]).argmax(-1)
        do = m.draft_model.model(input_ids=ids, use_cache=True, return_dict=True)
        k, v = M._extract_kv_lists(do.past_key_values)
        starts, n = m._exact_window_starts(do.last_hidden_state, k, v, th, lm, args.path_topk,
                                           chunk=32, all_positions=True)
    mask = lm[0, 1:].bool()
    mis = ((u != ids[0, 1:]) & mask).tolist()       # mis[t]: target argmax at t != data token t+1
    L = len(mis)
    path = dec = 0
    for s in starts[0].nonzero(as_tuple=True)[0].tolist():
        acc = int(n[0, s])
        path += any(mis[s + j] for j in range(max(acc - 1, 0)) if s + j < L)
        dec += any(mis[s + j] for j in range(acc) if s + j < L)
    mis_tot += sum(mis); sup_tot += int(mask.sum())
    win_tot += int(starts.sum()); path_tot += path; dec_tot += dec
    print(f"seq{i}: argmax != data at {sum(mis)}/{int(mask.sum())}, windows {int(starts.sum())}, "
          f"path leaves data {path}, decisive check leaves data {dec}")
print(f"total: argmax != data {mis_tot}/{sup_tot} ({mis_tot / sup_tot:.1%}); windows {win_tot}: "
      f"path {path_tot} ({path_tot / win_tot:.1%}), decisive {dec_tot} ({dec_tot / win_tot:.1%})")
