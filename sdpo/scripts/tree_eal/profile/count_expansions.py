"""How much of each sequence --exact_starts (Temporal, not --prob_starts) expands, in how many rounds.

Prints per sequence: supervised positions, real window starts, positions expanded, and the
positions expanded in each round. On 6 Qwen3-8B Alpaca regen sequences (A6000): starts are 27%
of the supervised positions, 55% get expanded, 29-36 rounds per sequence.
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
rounds = []
_ep = M.SmallLMTreeV2Model._expand_positions


def expand(self, rows, cols, *a, **k):
    rounds.append(len(rows))
    return _ep(self, rows, cols, *a, **k)


M.SmallLMTreeV2Model._expand_positions = expand
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
ds = build_dataset(tok, args.data, 2048, max_samples=args.n)
tot = {"sup": 0, "exp": 0, "starts": 0}
for i in range(len(ds)):
    ids = torch.tensor([ds[i]["input_ids"]]).cuda()
    lm = torch.tensor([ds[i]["loss_mask"]]).cuda()
    rounds.clear()
    with torch.no_grad():
        th = m.target_model.model(input_ids=ids, return_dict=True).last_hidden_state
        do = m.draft_model.model(input_ids=ids, use_cache=True, return_dict=True)
        keys, values = M._extract_kv_lists(do.past_key_values)
        starts, n = m._exact_window_starts(do.last_hidden_state, keys, values, th, lm,
                                           args.path_topk, chunk=64, all_positions=False)
    sup, st, ex = int(lm[:, 1:].sum()), int(starts.sum()), sum(rounds)
    tot["sup"] += sup; tot["exp"] += ex; tot["starts"] += st
    print(f"seq{i}: supervised {sup}, starts {st} (mean accept {float(n[starts].mean()):.2f}), "
          f"expanded {ex} ({ex / sup:.0%}), rounds {len(rounds)}, per round {rounds}")
print(f"total: supervised {tot['sup']}, starts {tot['starts']} ({tot['starts'] / tot['sup']:.0%}), "
      f"expanded {tot['exp']} ({tot['exp'] / tot['sup']:.0%})")
