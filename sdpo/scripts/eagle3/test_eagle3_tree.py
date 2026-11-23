"""GPU checks for the EAGLE-3 Tree-EAL port (one H200, ~30 min).

  python sdpo/scripts/eagle3/test_eagle3_tree.py --part {regress,llama,oss,all}

regress: refactored SmallLMTreeV2Model == the git HEAD version (same seed ->
         same loss and draft grads) on a 1B/1B pair, for several eal_modes.
llama:   Llama-3.1-8B + the released EAGLE3 head: root top-1 agreement with the
         target, tree-step logits == TTT-step logits, tree verify == plain
         forward, forward/backward of every supported mode.
oss:     gpt-oss-20b + a fresh reduced-vocab head: same checks; prefix > 128
         and branching trees exercise the sliding-window masks; bucket mass.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SDPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, SDPO)

from data_pipeline import DataCollator, build_dataset  # noqa: E402
from eagle3_tree_model import EAGLE3_TREE_MODES, Eagle3TreeModel  # noqa: E402
import small_lm_tree_v2_model as new_mod  # noqa: E402

L1B = "/scratch/yf3005/.huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
L8B = "/scratch/yf3005/models/Llama-3.1-8B-Instruct"
E3L = "/scratch/yf3005/models/EAGLE3-LLaMA3.1-Instruct-8B"
OSS = "/scratch/yf3005/.huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
DATA = "/scratch/yf3005/gto_data/llama31_8b_split/smoke200.jsonl"
DEV = "cuda"
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def batches(tokenizer, n, max_len=1024):
    ds = build_dataset(tokenizer, DATA, max_len, max_samples=n)
    col = DataCollator()
    return [{k: v.to(DEV) for k, v in col([ds[i]]).items()} for i in range(len(ds))]


KW = dict(anchor_kl_coef=0.1, eal_aux_coef=1.0, temperature=0.5, path_topk=4,
          n_anchors=2, n_draws=2, eal_score_coef=0.0)


def draft_grad_norm(model):
    g = [p.grad.float().norm() for p in model.draft_model.parameters() if p.grad is not None]
    return torch.stack(g).norm().item() if g else 0.0


# ---------------------------------------------------------------- regress

def _sync():
    if DEV != "cpu":
        torch.cuda.synchronize()


def _peak():
    return torch.cuda.max_memory_allocated() / 2**30 if DEV != "cpu" else 0.0


def _reset():
    if DEV != "cpu":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def part_regress(target=L1B, draft=L1B, dtype=torch.bfloat16, tok_path=None,
                 topk=(4, 3, 2, 1, 1, 1, 1), n_data=3, max_len=1024):
    print("== regress: refactor vs git HEAD", flush=True)
    old_src = subprocess.check_output(
        ["git", "-C", SDPO, "show", "HEAD:sdpo/small_lm_tree_v2_model.py"], text=True)
    path = "/scratch/yf3005/tmp/old_small_lm_tree_v2_model.py"
    with open(path, "w") as f:
        f.write(old_src)
    spec = importlib.util.spec_from_file_location("old_tree", path)
    old_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_mod)

    topk = list(topk)
    new = new_mod.SmallLMTreeV2Model(target, draft, topk, dtype=dtype).to(DEV)
    old = old_mod.SmallLMTreeV2Model.__new__(old_mod.SmallLMTreeV2Model)
    torch.nn.Module.__init__(old)
    for k in ("top_k_per_depth", "gamma", "tree_parents", "tree_depths", "tree_paths",
              "tree_depth_counts", "tree_depth_starts", "tree_n_nodes", "tree_budget"):
        setattr(old, k, getattr(new, k))
    old.target_model, old.draft_model = new.target_model, new.draft_model
    old.register_buffer("paths", new.paths.clone(), persistent=False)
    for m in (new, old):
        m.srsd_tgt_cache, m._srsd_tgt_bank = False, None
        m.train()
    tok = __import__("transformers").AutoTokenizer.from_pretrained(tok_path or target)
    data = batches(tok, n_data, max_len=max_len)
    for mode in ("sample_rsd", "sample_minpq", "soft_topk", "budget", "top4_kl_eal",
                 "greedy_topk", "path_topk"):
        res = []
        for m in (new, old):
            m.zero_grad(set_to_none=True)
            torch.manual_seed(1234)
            losses = []
            for b in data:
                loss, _, _ = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                               loss_mask=b["loss_mask"], eal_mode=mode, **KW)
                loss.backward()
                losses.append(loss.item())
            res.append((losses, draft_grad_norm(m)))
        (ln, gn), (lo, go) = res
        same = all(abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(ln, lo))
        check(f"regress {mode}", same and abs(gn - go) <= 1e-5 * max(1.0, go),
              f"new={['%.6f' % x for x in ln]} gn={gn:.6f} | old={['%.6f' % x for x in lo]} gn={go:.6f}")
    del new, old
    _reset()


# ---------------------------------------------------------------- shared EAGLE checks

def ttt_step_logits(model, b, steps):
    """Per-step draft logits [steps][B, L, Vd] via the TTT path (no grad)."""
    d = model.draft_model
    ids, am = b["input_ids"], b["attention_mask"]
    B, L = ids.shape
    with torch.no_grad():
        model._target_prefix(ids, am)
        aux, model._aux = model._aux, None
        nxt = torch.cat([ids[:, 1:], ids.new_zeros(B, 1)], 1)
        valid = torch.cat([am[:, 1:], am.new_zeros(B, 1)], 1)
        causal = model._draft_causal(valid, d.dtype)
        pos = torch.arange(L, device=DEV).view(1, L)
        hid = d.fc(aux.to(d.dtype))
        attn = d.midlayer.self_attn
        ks, vs, out = [], [], []
        for k in range(steps):
            q, kk, vv = attn.qkv(d.midlayer.inputs(d.embed(nxt), hid), pos + k)
            ks.append(kk)
            vs.append(vv)
            hid = d.midlayer.finish(hid, attn.attend_ttt(q, ks, vs, causal))
            out.append(d.logits(hid))
            nxt = torch.cat([nxt[:, 1:], nxt.new_zeros(B, 1)], 1)
    return out


def eagle_checks(model, data, tag, chain_model=None):
    d = model.draft_model
    to_d = None
    if d.reduced_vocab:
        lut = torch.full((d.t2d.numel(),), -1, dtype=torch.long, device=DEV)
        lut[model._t2d_ids()] = torch.arange(d.d2t.numel(), device=DEV)
        to_d = lambda t: lut[t]
    # (a) root top-1 agreement, and prefix logits == plain HF forward.
    agree = tot = 0
    for b in data:
        ids, am, lm = b["input_ids"], b["attention_mask"], b["loss_mask"].clone()
        lm[:, 0] = 0
        with torch.no_grad():
            tgt_h, _, _ = model._target_prefix(ids, am)
            drf_h, _, _ = model._draft_prefix(ids, am)
            a = lm[0].nonzero().view(-1)
            a = a[a < ids.size(1) - 1]
            t_arg = model.target_model.lm_head(tgt_h[0, a]).argmax(-1)
            d_arg = d.draft_to_target(d.logits(drf_h[0, a - 1]).argmax(-1))
            agree += (t_arg == d_arg).sum().item()
            tot += a.numel()
    check(f"{tag} root top-1 agreement", tot > 0, f"{agree / max(tot, 1):.3f} over {tot} anchors")
    b = max(data, key=lambda x: x["input_ids"].size(1))
    ids, am = b["input_ids"], b["attention_mask"]
    print(f"  longest sample: {ids.size(1)} tokens", flush=True)
    with torch.no_grad():
        ref = model.target_model(input_ids=ids).logits.float()
        tgt_h, tk, tv = model._target_prefix(ids, am)
        mine = model.target_model.lm_head(tgt_h).float()
    err = (ref - mine).abs().max().item() / ref.abs().max().item()
    check(f"{tag} target prefix logits == plain forward", err < 2e-2, f"rel max err {err:.2e}")

    # (c) tree verify == plain forward: chain of data tokens + one sibling.
    L = ids.size(1)
    # Latest anchor (past the sliding window) whose chain is in the draft vocab.
    a = max(1, min(300, L - 5))
    while to_d is not None and a > 1 and not bool((to_d(ids[:, a + 1:a + 4]) >= 0).all()):
        a -= 1
    chain = ids[:, a + 1:a + 4]
    chain_d = chain if to_d is None else to_d(chain)
    sib_d = (chain_d[:, :1] + 1) % d.lm_head.out_features
    sib = d.draft_to_target(sib_d)
    tree_t = torch.cat([chain, sib], 1)  # nodes: x_{a+1}, x_{a+2}, x_{a+3}, alt x_{a+1}
    par = torch.tensor([[-1, 0, 1, -1]], device=DEV)
    dep = torch.tensor([[0, 1, 2, 0]], device=DEV)
    tree_d = torch.cat([chain_d, sib_d], 1)
    ok_tok = bool((tree_d >= 0).all())
    if ok_tok:
        with torch.no_grad():
            got = model._target_tree_logits(tree_d, par, dep, [0], [a], tk, tv)
            seq = torch.cat([ids[:, :a + 1], chain], 1)
            ref_c = model._target_to_loss_vocab(model.target_model(input_ids=seq).logits.float()[:, a + 1:a + 4])
            seq_s = torch.cat([ids[:, :a + 1], sib], 1)
            ref_s = model._target_to_loss_vocab(model.target_model(input_ids=seq_s).logits.float()[:, a + 1])
        e1 = (got[:, :3] - ref_c).abs().max().item() / ref_c.abs().max().item()
        e2 = (got[:, 3] - ref_s).abs().max().item() / ref_s.abs().max().item()
        check(f"{tag} tree verify == plain forward (anchor {a})", max(e1, e2) < 2e-2,
              f"chain {e1:.2e} sibling {e2:.2e}")
    else:
        check(f"{tag} tree verify (tokens in draft vocab)", False, "chain token outside draft vocab")

    # (b) tree draft steps == TTT steps on a chain tree of data tokens.
    if chain_model is not None:
        cm = chain_model
        ttt = ttt_step_logits(cm, b, 3)
        with torch.no_grad():
            cm._target_prefix(ids, am)
            drf_h, dk, dv = cm._draft_prefix(ids, am)
            st = cm._draft_tree_begin([0], [a], dk, dv)
            root = cm._draft_to_loss_vocab(d.logits(drf_h[:, a - 1]))
            errs = [(root[..., :ttt[0].size(-1)] - ttt[0][:, a - 1]).abs().max().item()]
            for step in (1, 2):
                tok = ids[:, a + step:a + step + 1]
                tok = tok if to_d is None else to_d(tok)
                lg = cm._draft_tree_step(st, step, tok, use_cache=True)
                errs.append((lg[:, 0, :ttt[step].size(-1)] - ttt[step][:, a - 1]).abs().max().item())
        scale = ttt[0][:, a - 1].abs().max().item()
        check(f"{tag} tree draft steps == TTT steps", max(errs) / scale < 2e-2,
              f"rel errs {[f'{e / scale:.1e}' for e in errs]}")

    # bucket: loss-vocab softmax keeps the full-vocab probabilities.
    if d.reduced_vocab:
        T = model._temperature = 0.5
        z = ref[0, :64]
        lv = model._target_to_loss_vocab(z)
        p_full = torch.softmax(z / T, -1)
        p_lv = torch.softmax(lv / T, -1)
        e = (p_lv[:, :-1] - p_full[:, model._t2d_ids()]).abs().max().item()
        oov = 1 - p_full[:, model._t2d_ids()].sum(-1)
        eb = (p_lv[:, -1] - oov).abs().max().item()
        check(f"{tag} bucket keeps p exactly", max(e, eb) < 1e-4,
              f"in-vocab {e:.1e} bucket {eb:.1e} mean oov mass {oov.mean().item():.4f}")


def run_modes(model, data, tag, modes):
    model.train()
    model.srsd_tgt_cache, model._srsd_tgt_bank = False, None
    for mode in modes:
        _reset()
        _sync()
        t0 = time.time()
        try:
            vals = []
            for b in data:
                model.zero_grad(set_to_none=True)
                loss, _, m = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                                   loss_mask=b["loss_mask"], eal_mode=mode, **KW)
                loss.backward()
                vals.append((loss.item(), m["eal_mean"], m["kl_loss"], draft_grad_norm(model)))
            _sync()
            dt = (time.time() - t0) / len(data)
            fin = all(all(map(lambda x: x == x and abs(x) < 1e8, v)) for v in vals)
            check(f"{tag} {mode}", fin,
                  f"loss={vals[-1][0]:.4f} eal={sum(v[1] for v in vals) / len(vals):.3f} "
                  f"kl={vals[-1][2]:.4f} gn={vals[-1][3]:.3f} {dt:.2f}s/step "
                  f"peak={_peak():.1f}GiB")
        except Exception as e:  # keep going so one report covers every mode
            import traceback
            traceback.print_exc()
            check(f"{tag} {mode}", False, repr(e)[:300])
        _reset()


def build_pair(target, draft, topk, aux=None, dtype=torch.bfloat16):
    m = Eagle3TreeModel(target, draft, topk, dtype=dtype, aux_layers=aux).to(DEV)
    c = Eagle3TreeModel.__new__(Eagle3TreeModel)
    torch.nn.Module.__init__(c)
    c._init_tree([1, 1, 1], None)
    c.target_model, c.draft_model = m.target_model, m.draft_model
    c.sliding_window, c.aux_layers, c.ttt_length = m.sliding_window, m.aux_layers, m.ttt_length
    c._temperature, c._aux, c._drf_prefix_hidden = 1.0, None, None
    return m, c.to(DEV)


MODES = list(EAGLE3_TREE_MODES) + ["eagle3_ttt"]


def part_llama():
    print("== llama: Llama-3.1-8B + EAGLE3-LLaMA3.1-Instruct-8B", flush=True)
    m, c = build_pair(L8B, E3L, [4, 3, 2, 1, 1, 1, 1])
    m.eval()
    tok = __import__("transformers").AutoTokenizer.from_pretrained(L8B)
    data = batches(tok, 4)
    eagle_checks(m, data, "llama", chain_model=c)
    run_modes(m, data[:2], "llama", MODES)
    del m, c
    _reset()


def part_oss():
    print("== oss: gpt-oss-20b + fresh reduced-vocab head", flush=True)
    init = "/scratch/yf3005/tmp/eagle3_oss_test_init"
    if not os.path.isfile(os.path.join(init, "model.safetensors")):
        subprocess.check_call([sys.executable, os.path.join(HERE, "init_eagle3_draft.py"),
                               "--target", OSS, "--data", DATA, "--out", init,
                               "--count_samples", "200"])
    m, c = build_pair(OSS, init, [4, 3, 2, 1, 1, 1, 1])
    m.eval()
    tok = __import__("transformers").AutoTokenizer.from_pretrained(OSS)
    data = batches(tok, 4)
    print("  harmony sample:", repr(tok.decode(data[0]["input_ids"][0, :80])), flush=True)
    masked = data[0]["input_ids"][0][data[0]["loss_mask"][0].bool()]
    print("  loss-mask span:", repr(tok.decode(masked[:60])), flush=True)
    eagle_checks(m, data, "oss", chain_model=c)
    run_modes(m, data[:2], "oss", MODES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="all", choices=["regress", "llama", "oss", "all"])
    a = p.parse_args()
    torch.manual_seed(0)
    if a.part in ("regress", "all"):
        part_regress()
    if a.part in ("llama", "all"):
        part_llama()
    if a.part in ("oss", "all"):
        part_oss()
    print(f"== {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
