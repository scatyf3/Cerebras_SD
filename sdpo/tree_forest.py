"""Forest layout for the batched tree rollout (--forest).

The batch layout gives every tree its own row: the tree's prefix KV is copied
(index_select) and right-padded to the longest anchor, for the draft and the
target, and SDPA runs a dense mask over the whole padded prefix. The forest
layout puts all trees of the microbatch in ONE sequence instead:

  KV  = [prefix of seq 0 | prefix of seq 1 | ... ]  [depth-0 block | depth-1 block | ...]
        (each seq right-padded to max_kv)            (each block tree-major)

A node of tree b (sequence s_b, anchor a_b) sees prefix columns
s_b * max_kv + [0, a_b] and its own ancestors + itself. Same rows go through
the GEMMs; the prefix KV is shared (no per-tree copy), and FlexAttention with a
block mask skips the masked blocks instead of reading every padded column.
Logits are reshaped back to [B, n, V], so the losses do not change.
"""
import contextlib

import torch

# 64-row blocks skip more of the sparse forest than flex's default 128, and the
# tiles below fit in the ~100 KB shared memory of sm86 (A6000), where the
# default D=128 tiles do not compile.
BLOCK = 64
TILES = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 64,
         "BLOCK_M2": 64, "BLOCK_N2": 32}
# Below this many query rows SDPA with the dense mask is used. fp32 also stays
# on SDPA: these tiles need ~180 KB of shared memory in fp32 (sm86 has ~100 KB).
MIN_Q = 128
_flex = None


def pack_kv(keys, values, max_kv):
    """Per-layer [S, H, L, D] prefix KV → per-layer [1, H, S * max_kv, D]."""
    pk, pv = [], []
    for k, v in zip(keys, values):
        k, v = k[:, :, :max_kv], v[:, :, :max_kv]
        if k.size(0) > 1:
            k = k.transpose(0, 1).reshape(1, k.size(1), -1, k.size(3))
            v = v.transpose(0, 1).reshape(1, v.size(1), -1, v.size(3))
        pk.append(k); pv.append(v)
    return pk, pv


def pack_prefix(keys, values, max_kv, detach=False):
    """pack_kv as a DynamicCache the forest forwards append to."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for i, (k, v) in enumerate(zip(*pack_kv(keys, values, max_kv))):
        cache.update(k.detach() if detach else k, v.detach() if detach else v, i)
    return cache


def prefix_allowed(batch_idx, kv_lens, max_kv, n_seq, device):
    """[B, n_seq * max_kv] bool: tree b sees its own sequence up to its anchor."""
    seq = torch.tensor(batch_idx, device=device, dtype=torch.long)
    kv = torch.tensor(kv_lens, device=device, dtype=torch.long)
    own = seq.view(-1, 1) == torch.arange(n_seq, device=device).view(1, -1)
    upto = torch.arange(max_kv, device=device).view(1, -1) < kv.view(-1, 1)
    return (own.unsqueeze(-1) & upto.unsqueeze(1)).reshape(len(batch_idx), -1)


def block_diag(tree_allowed):
    """[B, q, k] per-tree allowed → [B*q, B*k], tree b only sees tree b."""
    B, q, k = tree_allowed.shape
    eye = torch.eye(B, dtype=torch.bool, device=tree_allowed.device)
    return (tree_allowed.unsqueeze(2) & eye.view(B, 1, B, 1)).reshape(B * q, B * k)


def forest_allowed(pre, tree_blocks, n_rows):
    """pre: [B, P] prefix allowed; tree_blocks: list of [B*n_rows, B*n_j] forest
    column blocks → [B*n_rows, P + sum B*n_j]."""
    rows = pre.unsqueeze(1).expand(-1, n_rows, -1).reshape(-1, pre.size(1))
    return torch.cat([rows] + list(tree_blocks), dim=1)


def forward(model, input_ids, position_ids, allowed, cache, use_cache):
    """One forest forward. allowed [Q, KV] bool → model output (logits [1, Q, V])."""
    dtype = model.dtype
    mask = torch.zeros(allowed.shape, device=allowed.device, dtype=dtype)
    mask = mask.masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]
    flex = (allowed.size(0) >= MIN_Q and allowed.is_cuda
            and dtype in (torch.bfloat16, torch.float16))
    with (_route_sdpa(_block_mask(allowed)) if flex else contextlib.nullcontext()):
        return model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                     past_key_values=cache, use_cache=use_cache, return_dict=True)


def _block_mask(allowed):
    from torch.nn.attention.flex_attention import create_block_mask

    def mask_mod(b, h, q, kv):
        return allowed[q, kv]

    Q, KV = allowed.shape
    return create_block_mask(mask_mod, 1, None, Q, KV, device=allowed.device,
                             BLOCK_SIZE=BLOCK)


def _flex_impl(query, key, value, block_mask, scale, enable_gqa):
    from torch.nn.attention.flex_attention import flex_attention
    return flex_attention(query, key, value, block_mask=block_mask, scale=scale,
                          enable_gqa=enable_gqa, kernel_options=TILES)


class _route_sdpa:
    """Routes 'sdpa' to FlexAttention with `block_mask` inside the block, for
    the calls whose (Q, KV) match it (every layer of the forest forward)."""

    def __init__(self, block_mask):
        self.bm = block_mask

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
        self.prev = prev = self.mapping["sdpa"]
        q_len, kv_len = self.bm.seq_lengths

        def attn(module, query, key, value, attention_mask, dropout=0.0,
                 scaling=None, is_causal=None, **kwargs):
            if query.size(2) != q_len or key.size(2) != kv_len:
                return prev(module, query, key, value, attention_mask, dropout=dropout,
                            scaling=scaling, is_causal=is_causal, **kwargs)
            global _flex
            if _flex is None:
                # dynamic=True: one compile serves every (Q, KV).
                _flex = torch.compile(_flex_impl, dynamic=True)
            scale = scaling if scaling is not None else query.size(-1) ** -0.5
            out = _flex(query, key, value, self.bm, scale, query.size(1) != key.size(1))
            return out.transpose(1, 2).contiguous(), None

        self.mapping["sdpa"] = attn

    def __exit__(self, *a):
        self.mapping["sdpa"] = self.prev
