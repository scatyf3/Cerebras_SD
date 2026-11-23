"""Tree-EAL training with an EAGLE-3 draft head (any HF target, incl. gpt-oss).

Eagle3TreeModel reuses SmallLMTreeV2Model's tree rollout, target verify and
every loss; it only overrides the model hooks (_target_prefix, _draft_prefix,
_*_anchor_logits, _draft_tree_begin/_step, _target_tree_logits, _node_kl).

EAGLE-3 alignment (same as traineagle3/cnets.py training and model/cnets.py
inference): draft slot t takes embed(x_{t+1}) and fc(target hidden h_t at the
aux layers), sits at RoPE position t and predicts x_{t+2}. For an anchor a
(prefix x_0..x_a, roots are candidates for x_{a+1}):
  * root logits  = draft output at slot a-1 (prefix keys: slots 0..a-1);
  * depth-d node = embed(node token) + the output hidden of its parent's step,
    at position a-1+d, attending the prefix slots, its ancestors and itself.
This is EAGLE-3's training-time test (TTT) step d at slot a-1 with the drafted
token in place of the data token, so tree logits and TTT logits coincide when
the tokens agree (tested in sdpo/scripts/eagle3/test_eagle3_tree.py).

Reduced draft vocabulary (d2t/t2d): losses run in a "loss vocabulary" of the
V_d draft tokens plus one bucket column holding all out-of-draft-vocab target
mass at the training temperature T (bucket = T * logsumexp(z_oov / T)). The
draft's bucket logit is OOV_LOGIT, so q puts no mass there. Everything the
losses compute at temperature T (alpha, min(p, q), residual TV, sampling) is
then exact. KL (temperature 1) drops the bucket: KL between p and q both
renormalised over the draft vocabulary, as EAGLE-3 trains.

eal_mode "eagle3_ttt" is plain EAGLE-3 training (TTT, soft CE against the
target over the draft vocab, step weights 0.8**k) for warm-starting a head.

Checkpoints use EAGLE-3 / SpecForge names (fc, midlayer.*, norm, lm_head, d2t,
t2d; no embed_tokens), so sglang can serve them as LlamaForCausalLMEagle3.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # pragma: no cover
    DynamicCache = None

try:
    from sdpo.small_lm_tree_v2_model import (
        SmallLMTreeV2Model, _extract_kv_lists, _pad_cache_batch)
except ImportError:
    from small_lm_tree_v2_model import (
        SmallLMTreeV2Model, _extract_kv_lists, _pad_cache_batch)

OOV_LOGIT = -1e4

# _rollout_batch modes whose losses only see (tokens, parents, drf_node,
# tgt_node) in the loss vocabulary. wsoft computes p at temperature 1 including
# the bucket, dense_tree / sample_rsd_offline need full-sequence draft logits or
# the offline bank; they stay small-LM only.
EAGLE3_TREE_MODES = ("budget", "path_topk", "node_marginal", "soft_topk",
                     "sample_minpq", "sample_rsd", "sample_anyk", "sample_cover",
                     "greedy_topk", "top4_kl_eal")
EAGLE3_MODES = EAGLE3_TREE_MODES + ("eagle3_ttt",)


class _RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg["hidden_size"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or H // self.n_heads
        self.rope_theta = float(cfg.get("rope_theta") or 10000.0)
        self.q_proj = nn.Linear(2 * H, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(2 * H, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(2 * H, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=False)

    def _rope(self, positions, dtype):
        """positions [B, n] -> cos, sin [B, 1, n, head_dim] (Llama rotate-half)."""
        d = self.head_dim
        inv = 1.0 / (self.rope_theta ** (
            torch.arange(0, d, 2, device=positions.device, dtype=torch.float32) / d))
        f = positions.float().unsqueeze(-1) * inv
        emb = torch.cat((f, f), dim=-1)
        return emb.cos().unsqueeze(1).to(dtype), emb.sin().unsqueeze(1).to(dtype)

    def qkv(self, x, positions):
        """x [B, n, 2H] -> q [B, nh, n, d], k, v [B, nkv, n, d], RoPE applied."""
        B, n, _ = x.shape
        q = self.q_proj(x).view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, n, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, n, self.n_kv, self.head_dim).transpose(1, 2)
        cos, sin = self._rope(positions, q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k, v

    def attend(self, q, k, v, mask):
        """Masked attention over explicit keys. mask: additive [B, 1, n, S]."""
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype),
                                             enable_gqa=self.n_heads != self.n_kv)
        B, _, n, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(B, n, -1))

    def attend_ttt(self, q, ks, vs, causal):
        """EAGLE-3 TTT attention for step len(ks)-1: causal over the step-0
        keys plus each later step's key at the same slot (the diagonal)."""
        g = self.n_heads // self.n_kv
        rep = (lambda t: t.repeat_interleave(g, dim=1)) if g > 1 else (lambda t: t)
        scale = self.head_dim ** -0.5
        k0, v0 = rep(ks[0]), rep(vs[0])
        scores = [torch.matmul(q, k0.transpose(-1, -2)) * scale + causal.to(q.dtype)]
        for kj in ks[1:]:
            scores.append(((q * rep(kj)).sum(-1) * scale).unsqueeze(-1))
        probs = torch.softmax(torch.cat(scores, dim=-1), dim=-1, dtype=torch.float32).to(q.dtype)
        L = k0.size(2)
        out = torch.matmul(probs[..., :L], v0)
        for j, vj in enumerate(vs[1:]):
            out = out + probs[..., L + j:L + j + 1] * rep(vj)
        B, _, n, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(B, n, -1))


class _MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, I = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MidLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, eps = cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-5)
        self.self_attn = _Attention(cfg)
        self.mlp = _MLP(cfg)
        self.hidden_norm = _RMSNorm(H, eps)
        self.input_layernorm = _RMSNorm(H, eps)
        self.post_attention_layernorm = _RMSNorm(H, eps)

    def inputs(self, emb, hid):
        return torch.cat((self.input_layernorm(emb), self.hidden_norm(hid)), dim=-1)

    def finish(self, hid, attn_out):
        h = hid + attn_out
        return h + self.mlp(self.post_attention_layernorm(h))


class Eagle3Draft(nn.Module):
    """One-layer EAGLE-3 head. The token embedding is the target's (frozen,
    not saved); call set_embedding before use."""

    def __init__(self, cfg):
        super().__init__()
        self.config = dict(cfg)
        H = cfg["hidden_size"]
        n_aux = int(cfg.get("num_aux_layers", 3))
        self.fc = nn.Linear(n_aux * int(cfg.get("target_hidden_size") or H), H, bias=False)
        self.midlayer = _MidLayer(cfg)
        self.norm = _RMSNorm(H, cfg.get("rms_norm_eps", 1e-5))
        V, Vd = int(cfg["vocab_size"]), int(cfg.get("draft_vocab_size") or cfg["vocab_size"])
        self.lm_head = nn.Linear(H, Vd, bias=False)
        self.register_buffer("d2t", torch.zeros(Vd, dtype=torch.long))
        self.register_buffer("t2d", torch.ones(V, dtype=torch.bool) if Vd == V
                             else torch.zeros(V, dtype=torch.bool))
        self._embed = None

    @property
    def dtype(self):
        return self.fc.weight.dtype

    @property
    def reduced_vocab(self):
        return self.lm_head.out_features < self.t2d.numel()

    def draft_to_target(self, tok):
        return tok + self.d2t[tok] if self.reduced_vocab else tok

    def set_embedding(self, embed):
        # Not a submodule: the target owns (and freezes) it.
        object.__setattr__(self, "_embed", embed)

    def embed(self, ids):
        return self._embed(ids).to(self.dtype)

    def step(self, emb, hid, positions, mask, past_k=None, past_v=None):
        """One draft pass over n new slots -> (out_hidden, k_new, v_new).
        mask: additive [B, 1, n, S] over past keys followed by the new ones."""
        attn = self.midlayer.self_attn
        q, k, v = attn.qkv(self.midlayer.inputs(emb, hid), positions)
        K = k if past_k is None else torch.cat([past_k, k], dim=2)
        V = v if past_v is None else torch.cat([past_v, v], dim=2)
        return self.midlayer.finish(hid, attn.attend(q, K, V, mask)), k, v

    def logits(self, hidden):
        return self.lm_head(self.norm(hidden)).float()

    # ---- checkpoint I/O (EAGLE-3 / SpecForge layout) ----

    @classmethod
    def from_pretrained(cls, path, dtype=torch.bfloat16):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        model = cls(cfg)
        sd = _load_state_dict(path)
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith("embed_tokens")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if not model.reduced_vocab:
            # Full draft vocab: identity maps, which the defaults already are.
            missing = [m for m in missing if m not in ("d2t", "t2d")]
        if missing or unexpected:
            raise RuntimeError(f"EAGLE-3 checkpoint {path}: missing={missing} "
                               f"unexpected={unexpected}")
        return model.to(dtype)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        cfg = dict(self.config)
        cfg.setdefault("architectures", ["LlamaForCausalLMEagle3"])
        cfg.setdefault("model_type", "llama")
        cfg["num_hidden_layers"] = 1
        cfg["torch_dtype"] = str(self.dtype).replace("torch.", "")
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        from safetensors.torch import save_file
        sd = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(sd, os.path.join(path, "model.safetensors"))


def _load_state_dict(path):
    st = os.path.join(path, "model.safetensors")
    if os.path.isfile(st):
        from safetensors.torch import load_file
        return load_file(st)
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(idx):
        from safetensors.torch import load_file
        with open(idx) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        sd = {}
        for fn in files:
            sd.update(load_file(os.path.join(path, fn)))
        return sd
    return torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu",
                      weights_only=True)


def default_aux_layers(n_layers):
    """hidden_states indices EAGLE-3 concatenates (traineagle3/cnets.py)."""
    return [2, n_layers // 2, n_layers - 3]


class Eagle3TreeModel(SmallLMTreeV2Model):

    def __init__(self, target_path, draft_path, top_k_per_depth,
                 dtype=torch.bfloat16, tree_budget=None, aux_layers=None, ttt_length=7):
        nn.Module.__init__(self)
        self._init_tree(top_k_per_depth, tree_budget)
        cfg = AutoConfig.from_pretrained(target_path)
        kw = {}
        attn = "eager" if getattr(cfg, "model_type", "") == "gpt_oss" else "sdpa"  # sinks
        qc = getattr(cfg, "quantization_config", None) or {}
        qm = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        if str(qm).lower().endswith("mxfp4"):
            # MXFP4 experts -> bf16, so the HF eager path runs without triton kernels.
            from transformers import Mxfp4Config
            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path, dtype=dtype, attn_implementation=attn, **kw)
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.eval()
        tcfg = self.target_model.config
        layer_types = getattr(tcfg, "layer_types", None) or []
        self.sliding_window = (int(tcfg.sliding_window)
                               if "sliding_attention" in layer_types and tcfg.sliding_window
                               else None)
        self.aux_layers = list(aux_layers or default_aux_layers(tcfg.num_hidden_layers))
        self.ttt_length = int(ttt_length)

        self.draft_model = Eagle3Draft.from_pretrained(draft_path, dtype)
        self.draft_model.set_embedding(self.target_model.get_input_embeddings())
        if self.draft_model.config["vocab_size"] != tcfg.vocab_size:
            raise ValueError(f"draft vocab_size {self.draft_model.config['vocab_size']} "
                             f"!= target {tcfg.vocab_size}")
        if self.draft_model.fc.in_features != len(self.aux_layers) * tcfg.hidden_size:
            raise ValueError("draft fc input does not match aux_layers x target hidden size")
        self._temperature = 1.0
        self._aux = None
        self._drf_prefix_hidden = None

    # ---- vocabularies ----

    def _t2d_ids(self):
        d = self.draft_model
        return torch.arange(d.d2t.numel(), device=d.d2t.device) + d.d2t

    def _target_to_loss_vocab(self, logits):
        """Target logits over the full vocab -> loss vocab (draft vocab + bucket)."""
        if not self.draft_model.reduced_vocab:
            return logits
        T = self._temperature
        z_in = logits[..., self._t2d_ids()]
        lse_all = torch.logsumexp(logits / T, dim=-1)
        lse_in = torch.logsumexp(z_in / T, dim=-1)
        frac_in = torch.exp(lse_in - lse_all).clamp(max=1.0 - 1e-7)
        bucket = (T * (lse_all + torch.log1p(-frac_in))).clamp(min=OOV_LOGIT)
        return torch.cat([z_in, bucket.unsqueeze(-1)], dim=-1)

    def _draft_to_loss_vocab(self, logits):
        if not self.draft_model.reduced_vocab:
            return logits
        return torch.cat([logits, logits.new_full(logits.shape[:-1] + (1,), OOV_LOGIT)], dim=-1)

    def _node_kl(self, tgt_node, drf_node):
        if self.draft_model.reduced_vocab:
            tgt_node, drf_node = tgt_node[..., :-1], drf_node[..., :-1]
        return SmallLMTreeV2Model._node_kl(tgt_node, drf_node)

    # ---- masks ----

    def _causal_masks(self, attention_mask, dtype):
        """Prefix masks: additive [B, 1, L, L]; a dict per layer type when the
        target has sliding-window layers (gpt-oss), since a 4D mask is otherwise
        used as-is by every layer."""
        B, L = attention_mask.shape
        dev = attention_mask.device
        i = torch.arange(L, device=dev)
        allowed = (i.view(1, L, 1) >= i.view(1, 1, L)) & attention_mask.bool().view(B, 1, L)
        neg = torch.finfo(dtype).min
        full = torch.zeros(B, 1, L, L, device=dev, dtype=dtype).masked_fill_(~allowed.unsqueeze(1), neg)
        if self.sliding_window is None:
            return full
        far = (i.view(L, 1) - i.view(1, L)) >= self.sliding_window
        return {"full_attention": full,
                "sliding_attention": full.masked_fill(far.view(1, 1, L, L), neg)}

    # ---- hooks ----

    def _target_prefix(self, input_ids, attention_mask):
        # Explicit DynamicCache(): without a config every layer keeps the full
        # prefix (a config-built cache keeps only the sliding window).
        out = self.target_model.model(
            input_ids=input_ids,
            attention_mask=self._causal_masks(attention_mask, self.target_model.dtype),
            past_key_values=DynamicCache(), use_cache=True,
            output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        self._aux = torch.cat([hs[i] for i in self.aux_layers], dim=-1)
        keys, values = _extract_kv_lists(out.past_key_values)
        return out.last_hidden_state, keys, values

    def _draft_prefix(self, input_ids, attention_mask):
        """Draft over every slot t: embed(x_{t+1}) + fc(h_t) at position t."""
        aux, self._aux = self._aux, None
        d = self.draft_model
        B, L = input_ids.shape
        nxt = torch.cat([input_ids[:, 1:], input_ids.new_zeros(B, 1)], dim=1)
        valid = torch.cat([attention_mask[:, 1:], attention_mask.new_zeros(B, 1)], dim=1)
        pos = torch.arange(L, device=input_ids.device).view(1, L).expand(B, L)
        mask = self._draft_causal(valid, d.dtype)
        out, k, v = d.step(d.embed(nxt), d.fc(aux.to(d.dtype)), pos, mask)
        self._drf_prefix_hidden = out
        return out, [k], [v]

    @staticmethod
    def _draft_causal(valid, dtype):
        B, L = valid.shape
        i = torch.arange(L, device=valid.device)
        allowed = (i.view(1, L, 1) >= i.view(1, 1, L)) & valid.bool().view(B, 1, L)
        return torch.zeros(B, 1, L, L, device=valid.device, dtype=dtype).masked_fill_(
            ~allowed.unsqueeze(1), torch.finfo(dtype).min)

    def _target_anchor_logits(self, hidden, batch_idx, anchor_pos):
        rows = self._anchor_rows(hidden, batch_idx, anchor_pos)
        return self._target_to_loss_vocab(self.target_model.lm_head(rows).float())

    def _draft_anchor_logits(self, hidden, batch_idx, anchor_pos):
        rows = self._anchor_rows(hidden, batch_idx, [a - 1 for a in anchor_pos])
        return self._draft_to_loss_vocab(self.draft_model.logits(rows))

    def _draft_tree_begin(self, batch_idx, anchor_pos, drf_keys, drf_values):
        kv_lens = list(anchor_pos)  # slots 0..a-1
        max_kv = max(kv_lens)
        dev = drf_keys[0].device
        b_t = torch.tensor(batch_idx, device=dev, dtype=torch.long)
        a_t = torch.tensor(anchor_pos, device=dev, dtype=torch.long)
        return {"k": drf_keys[0][:, :, :max_kv].index_select(0, b_t),
                "v": drf_values[0][:, :, :max_kv].index_select(0, b_t),
                "hid": self._drf_prefix_hidden[b_t, a_t - 1].unsqueeze(1),
                "kv_lens": kv_lens, "max_kv": max_kv, "a_t": a_t}

    def _draft_tree_step(self, state, d_step, tokens, use_cache):
        d = self.draft_model
        B, n_in = tokens.shape
        par = state["hid"]
        if par.size(1) != n_in:
            par = par.repeat_interleave(n_in // par.size(1), dim=1)
        pos = (state["a_t"] - 1 + d_step).view(B, 1).expand(B, n_in)
        mask = self._draft_step_mask_batch(
            d_step, state["kv_lens"], state["max_kv"], tokens.device, d.dtype)
        out, k, v = d.step(d.embed(d.draft_to_target(tokens)), par, pos, mask,
                           state["k"], state["v"])
        state["k"] = torch.cat([state["k"], k], dim=2)
        state["v"] = torch.cat([state["v"], v], dim=2)
        state["hid"] = out
        return self._draft_to_loss_vocab(d.logits(out))

    def _target_tree_logits(self, tokens, parents, depths, batch_idx, anchor_pos,
                            tgt_keys, tgt_values):
        B, N = tokens.shape
        device = tokens.device
        dtype = self.target_model.dtype
        kv_lens = [ap + 1 for ap in anchor_pos]
        a_t = torch.tensor(anchor_pos, device=device, dtype=torch.long)
        cache, max_kv = _pad_cache_batch(tgt_keys, tgt_values, batch_idx, kv_lens)
        if depths.dim() == 1:
            depths = depths.view(1, -1)
        pos = (a_t.view(B, 1) + 1 + depths).expand(B, N)
        mask = self._target_mask_batch(kv_lens, max_kv, parents, device, dtype)
        if self.sliding_window is not None:
            k_pos = torch.cat([torch.arange(max_kv, device=device).view(1, -1).expand(B, -1),
                               pos], dim=1)
            far = (pos.unsqueeze(-1) - k_pos.unsqueeze(1)) >= self.sliding_window
            mask = {"full_attention": mask,
                    "sliding_attention": mask.masked_fill(far.unsqueeze(1), torch.finfo(dtype).min)}
        with torch.no_grad():
            out = self.target_model(
                input_ids=self.draft_model.draft_to_target(tokens),
                attention_mask=mask, position_ids=pos, past_key_values=cache,
                use_cache=False, return_dict=True)
        return self._target_to_loss_vocab(out.logits.float())

    # ---- forward ----

    def forward(self, input_ids, attention_mask, loss_mask, *args, **kwargs):
        eal_mode = kwargs.get("eal_mode", "budget")
        if eal_mode not in EAGLE3_MODES:
            raise ValueError(f"eal_mode {eal_mode!r} is not supported with an EAGLE-3 "
                             f"draft; use one of {EAGLE3_MODES}")
        for flag in ("dense_eal_start", "dense_kl", "exact_starts", "top4path_starts"):
            if getattr(self, flag, False):
                raise ValueError(f"--{flag} is not supported with an EAGLE-3 draft")
        if float(getattr(self, "gto_coef", 0.0)) > 0:
            raise ValueError("--gto_coef is not supported with an EAGLE-3 draft")
        self._temperature = max(float(kwargs.get("temperature", 1.0)), 1e-6)
        # Slot a-1 has to exist for the root logits.
        loss_mask = loss_mask.clone()
        loss_mask[:, 0] = 0
        try:
            if eal_mode == "eagle3_ttt":
                return self._forward_ttt(input_ids, attention_mask, loss_mask)
            return super().forward(input_ids, attention_mask, loss_mask, *args, **kwargs)
        finally:
            self._aux = None
            self._drf_prefix_hidden = None

    def _forward_ttt(self, input_ids, attention_mask, loss_mask):
        """Plain EAGLE-3 training: ttt_length steps of soft CE over the draft vocab."""
        d = self.draft_model
        B, L = input_ids.shape
        dev = input_ids.device
        with torch.no_grad():
            tgt_hidden, _, _ = self._target_prefix(input_ids, attention_mask)
            full = self.target_model.lm_head(tgt_hidden)
            # tgt[t] = target logits at t+1 (the token after x_{t+1}), as cnets pads.
            tgt_in = full[:, 1:, self._t2d_ids()].float()
            tgt_p = torch.softmax(tgt_in, dim=-1)
            in_vocab = d.t2d[full[:, 1:].argmax(-1)]
            del full, tgt_hidden, tgt_in
        aux, self._aux = self._aux, None
        pad1 = lambda t: torch.cat([t, t.new_zeros((B, 1) + t.shape[2:])], dim=1)
        tgt_p, in_vocab = pad1(tgt_p), pad1(in_vocab)
        lm = loss_mask.to(tgt_p.dtype)

        ids = torch.cat([input_ids[:, 1:], input_ids.new_zeros(B, 1)], dim=1)
        valid = torch.cat([attention_mask[:, 1:], attention_mask.new_zeros(B, 1)], dim=1)
        causal = self._draft_causal(valid, d.dtype)
        base_pos = torch.arange(L, device=dev).view(1, L).expand(B, L)
        hid = d.fc(aux.to(d.dtype))
        attn = d.midlayer.self_attn
        ks, vs = [], []
        total = hid.new_zeros((), dtype=torch.float32)
        step_loss, step_acc = [], []
        for k in range(self.ttt_length):
            q, kk, vv = attn.qkv(d.midlayer.inputs(d.embed(ids), hid), base_pos + k)
            ks.append(kk)
            vs.append(vv)
            hid = d.midlayer.finish(hid, attn.attend_ttt(q, ks, vs, causal))
            logp = F.log_softmax(d.logits(hid), dim=-1)
            w = lm * in_vocab.to(lm.dtype)
            loss_k = -(w * (tgt_p * logp).sum(-1)).sum() / w.sum().clamp(min=1.0)
            total = total + (0.8 ** k) * loss_k
            with torch.no_grad():
                acc = ((logp.argmax(-1) == tgt_p.argmax(-1)).float() * w).sum() / w.sum().clamp(min=1.0)
            step_loss.append(loss_k.detach())
            step_acc.append(acc)
            ids = torch.cat([ids[:, 1:], ids.new_zeros(B, 1)], dim=1)
            tgt_p, in_vocab = pad1(tgt_p[:, 1:]), pad1(in_vocab[:, 1:])
            lm = torch.cat([lm[:, 1:], lm.new_zeros(B, 1)], dim=1)

        losses = torch.stack(step_loss).float()
        accs = torch.stack(step_acc).float()
        chain_al = torch.cumprod(accs, dim=0).sum()
        vals = torch.cat([total.detach().view(1), losses, accs, chain_al.view(1)]).tolist()
        n = self.ttt_length
        metrics = {
            "total_loss": vals[0], "kl_loss": vals[1], "aux_loss": -vals[-1],
            "eal_mean": vals[-1], "eal_loss": 0.0, "dense_loss": vals[0],
            "num_anchors": int(loss_mask.sum().item()), "num_valid": float(loss_mask.sum().item()),
            "mean_tau": vals[-1], "eagle_loss": vals[1], "tree_nodes": 0,
        }
        for k in range(n):
            metrics[f"ttt{k}_loss"] = vals[1 + k]
            metrics[f"ttt{k}_acc"] = vals[1 + n + k]
        return total, total.detach(), metrics
