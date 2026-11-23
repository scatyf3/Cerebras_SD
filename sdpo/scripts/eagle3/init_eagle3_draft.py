"""Create a fresh EAGLE-3 draft head directory for a target model.

  python sdpo/scripts/eagle3/init_eagle3_draft.py \
      --target /path/to/gpt-oss-20b --data train.jsonl --out /path/to/draft_init

Draft vocabulary: the --draft_vocab_size most frequent target tokens under the
training loss mask (EAGLE-3's scandata), plus every special token. 0 keeps the
full vocabulary. lm_head starts from the target's lm_head rows for those
tokens; fc and the decoder layer are N(0, 0.02), norms are 1. Warm-start the
head with --eal_mode eagle3_ttt before Tree-EAL training.
"""

import argparse
import json
import os
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from data_pipeline import build_dataset  # noqa: E402
from eagle3_tree_model import Eagle3Draft  # noqa: E402


def target_lm_head(path):
    from safetensors import safe_open
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx) as f:
            fn = json.load(f)["weight_map"]["lm_head.weight"]
    else:
        fn = "model.safetensors"
    with safe_open(os.path.join(path, fn), framework="pt") as f:
        return f.get_tensor("lm_head.weight")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--data", required=True, help="training jsonl (token counts)")
    p.add_argument("--out", required=True)
    p.add_argument("--draft_vocab_size", type=int, default=32000)
    p.add_argument("--count_samples", type=int, default=20000)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--num_attention_heads", type=int, default=0, help="0 = target's")
    p.add_argument("--num_key_value_heads", type=int, default=0, help="0 = target's")
    p.add_argument("--head_dim", type=int, default=0, help="0 = target's")
    p.add_argument("--intermediate_size", type=int, default=0,
                   help="0 = 6 x hidden_size")
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    tcfg = AutoConfig.from_pretrained(args.target)
    tok = AutoTokenizer.from_pretrained(args.target)
    H, V = tcfg.hidden_size, tcfg.vocab_size

    if args.draft_vocab_size and args.draft_vocab_size < V:
        ds = build_dataset(tok, args.data, args.max_len, max_samples=args.count_samples)
        counts = Counter()
        for ids, m in zip(ds["input_ids"], ds["loss_mask"]):
            counts.update(t for t, k in zip(ids, m) if k)
        total = sum(counts.values())
        special = set(tok.all_special_ids)
        special |= {i for t, i in tok.get_vocab().items() if t.startswith("<|") and t.endswith("|>")}
        special = {i for i in special if i < V}
        used = set(special)
        for t, _ in counts.most_common():
            if len(used) >= args.draft_vocab_size:
                break
            used.add(t)
        used = sorted(used)[:args.draft_vocab_size]
        covered = sum(counts[t] for t in used) / max(total, 1)
        print(f"draft vocab {len(used)} of {V}: covers {covered:.2%} of "
              f"{total} loss-mask tokens ({len(ds)} samples)")
        Vd = len(used)
    else:
        used, Vd = list(range(V)), V
        print(f"full draft vocab {V}")

    cfg = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "hidden_size": H,
        "target_hidden_size": H,
        "num_aux_layers": 3,
        "intermediate_size": args.intermediate_size or 6 * H,
        "num_attention_heads": args.num_attention_heads or tcfg.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads or tcfg.num_key_value_heads,
        "head_dim": args.head_dim or getattr(tcfg, "head_dim", None) or H // tcfg.num_attention_heads,
        "num_hidden_layers": 1,
        "hidden_act": "silu",
        "rms_norm_eps": float(getattr(tcfg, "rms_norm_eps", 1e-5)),
        "rope_theta": args.rope_theta,
        "max_position_embeddings": args.max_len * 2,
        "vocab_size": V,
        "draft_vocab_size": Vd,
        "tie_word_embeddings": False,
        "eagle_target": os.path.abspath(args.target),
    }
    torch.manual_seed(args.seed)
    model = Eagle3Draft(cfg)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight, std=0.02)
    used_t = torch.tensor(used, dtype=torch.long)
    with torch.no_grad():
        model.lm_head.weight.copy_(target_lm_head(args.target).float()[used_t])
        if Vd < V:
            model.d2t.copy_(used_t - torch.arange(Vd))
            model.t2d.zero_()
            model.t2d[used_t] = True
    model.to(torch.bfloat16).save_pretrained(args.out)
    print(f"wrote {args.out}: {json.dumps({k: cfg[k] for k in ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'draft_vocab_size')})}")


if __name__ == "__main__":
    main()
