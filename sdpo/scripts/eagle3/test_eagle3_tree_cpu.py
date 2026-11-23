"""CPU version of test_eagle3_tree.py on tiny random models (minutes, no GPU).

Random weights make the accept-length numbers meaningless, but every
equivalence check (refactor regression, tree verify vs plain forward with a
sliding window of 8, tree steps vs TTT, bucket mass) is exact in float32.
"""

import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_eagle3_tree as T  # noqa: E402

T.DEV = "cpu"
# Login nodes allow 3 GiB per user: one small tree per step.
T.KW.update(n_anchors=1, n_draws=1, path_topk=2)
ROOT = os.environ.get("TINY_ROOT", "/scratch/yf3005/tmp/eagle3_tiny")


def make_tiny():
    from transformers import (AutoTokenizer, GptOssConfig, GptOssForCausalLM,
                              LlamaConfig, LlamaForCausalLM)
    paths = {k: os.path.join(ROOT, k) for k in ("llama_t", "llama_d", "oss_t", "llama_e3", "oss_e3")}
    if not os.path.isfile(os.path.join(paths["llama_t"], "config.json")):
        tok = AutoTokenizer.from_pretrained(T.L1B)
        for name, layers, seed in (("llama_t", 6, 0), ("llama_d", 2, 1)):
            torch.manual_seed(seed)
            cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                              num_hidden_layers=layers, num_attention_heads=4,
                              num_key_value_heads=2, max_position_embeddings=4096,
                              tie_word_embeddings=False, bos_token_id=tok.bos_token_id,
                              eos_token_id=tok.eos_token_id)
            LlamaForCausalLM(cfg).save_pretrained(paths[name])
            tok.save_pretrained(paths[name])
    if not os.path.isfile(os.path.join(paths["oss_t"], "config.json")):
        tok = AutoTokenizer.from_pretrained(T.OSS)
        torch.manual_seed(2)
        cfg = GptOssConfig(vocab_size=201088, hidden_size=64, intermediate_size=64,
                           num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=16, num_local_experts=4, num_experts_per_tok=2,
                           sliding_window=8, max_position_embeddings=4096,
                           layer_types=["sliding_attention", "full_attention"] * 3,
                           rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0,
                                         "beta_slow": 1.0, "truncate": False,
                                         "original_max_position_embeddings": 4096},
                           tie_word_embeddings=False)
        m = GptOssForCausalLM(cfg)
        with torch.no_grad():  # sinks start at zero; make them matter
            for layer in m.model.layers:
                layer.self_attn.sinks.normal_(0, 1.0)
        m.save_pretrained(paths["oss_t"])
        tok.save_pretrained(paths["oss_t"])
    init = os.path.join(HERE, "init_eagle3_draft.py")
    for tgt, out, vd in (("llama_t", "llama_e3", "0"), ("oss_t", "oss_e3", "3000")):
        if not os.path.isfile(os.path.join(paths[out], "model.safetensors")):
            subprocess.check_call([sys.executable, init, "--target", paths[tgt], "--data", T.DATA,
                                   "--out", paths[out], "--draft_vocab_size", vd,
                                   "--count_samples", "200", "--intermediate_size", "128"])
    return paths


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    torch.manual_seed(0)
    p = make_tiny()
    from transformers import AutoTokenizer
    if not os.environ.get("SKIP_REGRESS"):
        T.part_regress(target=p["llama_t"], draft=p["llama_d"], dtype=torch.float32,
                       topk=(3, 2, 1, 1), n_data=2, max_len=200)
    for tag, tgt, e3 in (("tiny-llama", "llama_t", "llama_e3"), ("tiny-oss", "oss_t", "oss_e3")):
        print(f"== {tag}", flush=True)
        m, c = T.build_pair(p[tgt], p[e3], [3, 2, 1, 1], dtype=torch.float32)
        m.eval()
        data = T.batches(AutoTokenizer.from_pretrained(p[tgt]), 2, max_len=200)
        T.eagle_checks(m, data, tag, chain_model=c)
        T.run_modes(m, data[:2], tag, T.MODES)
        del m, c
    print(f"== {'ALL PASS' if not T.FAILS else 'FAILED: ' + ', '.join(T.FAILS)}", flush=True)
    sys.exit(1 if T.FAILS else 0)


if __name__ == "__main__":
    main()
