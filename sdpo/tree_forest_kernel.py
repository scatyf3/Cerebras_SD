"""Triton forest attention for the no-grad CUDA-graph rollout (--rollout graph).

From FastDraftRL (src/forest_attn.py): a tree spec-decode attention kernel with
runtime Q_LEN / MAX_KV, an optional GPU-scalar seqlen (skips the unused tail of
the static KV cache) and a 64x64 visibility bitmap (skips fully masked tiles).
Measured vs SDPA with a forest mask (Hq=16, Hkv=8, D=128, bf16): Q=384 0.11 vs
0.52 ms, Q=1664 0.64 vs 5.47 ms per layer.

tree_graph_rollout routes 'sdpa' here (route_sdpa) only while it captures, so
the draft model keeps attn_implementation='sdpa' everywhere else.
"""
import torch
import triton
import triton.language as tl

ATTN_IMPL = "forest_triton"


def _configs():
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
            for bm in (32, 64) for bn in (32, 64) for nw in (4, 8) for ns in (2, 3)]


@triton.autotune(configs=_configs(), key=["Q_BUCKET", "KV_BUCKET", "BLOCK_D", "GROUP"])
@triton.jit(do_not_specialize=["Q_LEN", "MAX_KV"])
def _forest_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, Mask_ptr, Seqlen_ptr, Bmap_ptr, bmap_stride, scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_mb, stride_mm, stride_mn,
    Q_LEN, MAX_KV, Q_BUCKET, KV_BUCKET,
    GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr, HAS_SEQLEN: tl.constexpr, HAS_BMAP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    kv_h = pid_h // GROUP
    # Valid KV = [0, kv_end). With a GPU-scalar seqlen (CUDA-graph rollout over a
    # preallocated StaticCache) the walk skips the unused tail without a host sync.
    kv_end = MAX_KV
    if HAS_SEQLEN:
        kv_end = tl.minimum(tl.load(Seqlen_ptr).to(tl.int32), MAX_KV)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(Q_ptr + pid_b * stride_qb + pid_h * stride_qh
                + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < Q_LEN, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    mask_row = Mask_ptr + pid_b * stride_mb + offs_m[:, None] * stride_mm

    bmap_row = Bmap_ptr + ((pid_m * BLOCK_M) // 64) * bmap_stride
    for start_n in range(0, MAX_KV, BLOCK_N):
        live = start_n < kv_end
        if HAS_BMAP:
            # 64x64 "any visible" bitmap of the mask (BLOCK_M/N <= 64, so every
            # tile lies inside one bitmap cell): skip fully masked tiles.
            live = live & (tl.load(bmap_row + start_n // 64) != 0)
        if live:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_ok = offs_n[:, None] < kv_end
            k = tl.load(K_ptr + pid_b * stride_kb + kv_h * stride_kh
                        + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                        mask=kv_ok, other=0.0)
            qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
            qk += tl.load(mask_row + offs_n[None, :] * stride_mn,
                          mask=(offs_m[:, None] < Q_LEN) & (offs_n[None, :] < kv_end),
                          other=float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)     # fully-masked rows
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(qk - m_safe[:, None])
            v = tl.load(V_ptr + pid_b * stride_vb + kv_h * stride_vh
                        + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                        mask=kv_ok, other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

    acc = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    tl.store(Out_ptr + pid_b * stride_ob + pid_h * stride_oh
             + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(Out_ptr.dtype.element_ty), mask=offs_m[:, None] < Q_LEN)


def _pow2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


def forest_attention(query, key, value, mask, scale, seqlen=None, bmap=None):
    """query [B,Hq,Q,D]; key/value [B,Hkv,KV,D]; mask additive [B,Q,KV] fp32;
    seqlen: optional int32 [1] GPU scalar bounding the valid KV prefix;
    bmap: optional uint8 [ceil(Q/64), KV/64], nonzero where the 64x64 mask
    block has a visible entry (B = 1)."""
    B, Hq, Q, D = query.shape
    Hkv, KV = key.shape[1], key.shape[2]
    assert Hq % Hkv == 0 and mask.shape[-2:] == (Q, KV)
    out = torch.empty_like(query)
    grid = lambda meta: (triton.cdiv(Q, meta["BLOCK_M"]), Hq, B)  # noqa: E731
    _forest_attn_kernel[grid](
        query, key, value, out, mask, mask if seqlen is None else seqlen,
        mask if bmap is None else bmap, 0 if bmap is None else bmap.stride(0), float(scale),
        *query.stride(), *key.stride(), *value.stride(), *out.stride(), *mask.stride(),
        Q, KV, _pow2(Q), _pow2(KV),
        GROUP=Hq // Hkv, BLOCK_D=D, HAS_SEQLEN=seqlen is not None, HAS_BMAP=bmap is not None,
    )
    return out


# Set by GraphRolloutEngine around warmup / capture so the captured kernels read
# its GPU-scalar seqlen buffer and the bucket's block bitmap. None elsewhere.
_SEQLEN = None
_BMAP = None


def set_seqlen(t, bmap=None):
    global _SEQLEN, _BMAP
    _SEQLEN, _BMAP = t, bmap


class route_sdpa:
    """Inside the block, forest calls of 'sdpa' (4D float mask, Q > 1, no grad)
    run the Triton kernel with the current seqlen / bitmap; others are unchanged."""

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
        self.prev = prev = self.mapping["sdpa"]

        def attn(module, query, key, value, attention_mask, dropout=0.0,
                 scaling=None, is_causal=None, **kwargs):
            forest = (attention_mask is not None and attention_mask.ndim == 4
                      and attention_mask.is_floating_point() and query.shape[2] > 1
                      and attention_mask.shape[-1] == key.shape[-2]
                      and not (torch.is_grad_enabled() and query.requires_grad))
            if not forest:
                return prev(module, query, key, value, attention_mask, dropout=dropout,
                            scaling=scaling, is_causal=is_causal, **kwargs)
            scale = scaling if scaling is not None else query.shape[-1] ** -0.5
            add = attention_mask[:, 0]
            if add.dtype != torch.float32 or not add.is_contiguous():
                add = add.float().contiguous()
            bmap = _BMAP if _BMAP is not None and _BMAP.shape[0] * 64 >= add.shape[1] else None
            out = forest_attention(query, key, value, add, scale, seqlen=_SEQLEN, bmap=bmap)
            return out.transpose(1, 2).contiguous(), None

        self.mapping["sdpa"] = attn

    def __exit__(self, *a):
        self.mapping["sdpa"] = self.prev
