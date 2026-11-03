"""CUDA-graph replay of the draft's per-depth forest forward during rollout
(--rollout graph). From FastDraftRL (src/graph_rollout.py).

Every rollout forward has the same structure (n leaf tokens on top of
prefix + already-placed forest nodes), so it is captured once per Q bucket and
replayed with new inputs:

  * KV lives in a preallocated `StaticCache` of `max_kv` slots. At the start of
    each rollout the draft prefix KV (from the prefix forward) is copied into
    slots [0, L). Forest node j lives at slot L + j, exactly as in the eager
    DynamicCache path.
  * The write position is the StaticLayer's own GPU `cumulative_length`
    (index_copy_), set in place before each replay.
  * Inputs are padded to the bucket. Padded rows are fully masked and their KV
    lands in slots after the real rows, which later writes overwrite and the
    mask never exposes.
  * The additive mask is a static fp32 buffer [1, 1, Q_bucket, max_kv]. The
    captured attention is the Triton forest kernel (tree_forest_kernel), whose
    GPU-scalar seqlen skips the unused KV tail.

Graphs read the parameters by address, so in-place optimizer updates (torch
AdamW, DeepSpeed ZeRO-0/1/2) are picked up by later replays. If any parameter's
storage moves, all graphs are dropped and recaptured. ZeRO-3 gathers
parameters on the fly and cannot be captured; callers fall back to eager.
"""
import torch
from transformers.cache_utils import StaticCache

import tree_forest_kernel as forest_attn


def _pow2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


def _bucket(n, min_q):
    """Q bucket for n rows: powers of 2 up to 64, then multiples of 64. The
    default tree's per-depth leaf counts are 64/192/384; pow2 buckets padded
    them to 64/256/512 (24% wasted rows over a rollout)."""
    if n <= 64:
        return max(min_q, _pow2(n))
    return -(-n // 64) * 64


def graph_capturable(model):
    """False under DeepSpeed ZeRO-3 (partitioned params have ds_tensor)."""
    return not any(hasattr(p, "ds_tensor") for p in model.parameters())


class _Bucket:
    def __init__(self, q, max_kv, device, mask_dtype):
        self.q = q
        self.ids = torch.zeros(1, q, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, q, dtype=torch.long, device=device)
        self.mask = torch.zeros(1, 1, q, max_kv, dtype=mask_dtype, device=device)
        # 64x64 "any visible" bitmap of mask, read by the Triton kernel to skip
        # fully masked tiles (rows past the anchor's prefix, non-ancestor nodes).
        self.bmap = torch.zeros(-(-q // 64), max_kv // 64, dtype=torch.uint8, device=device)
        self.graph = None
        self.logits = None


class GraphRolloutEngine:

    def __init__(self, model, max_kv, min_q=16, forward=None):
        self.model = model
        # Callable captured into the graphs (default: the module). A
        # torch.compile'd model.forward fuses the ~450 small HF kernels of a
        # forward; it specializes per bucket, so each bucket compiles once.
        self.fwd = forward or model
        self.max_kv = int(max_kv)
        assert self.max_kv % 64 == 0, "max_kv must be a multiple of 64 (block bitmap)"
        self.min_q = min_q
        p = next(model.parameters())
        self.device = p.device
        # The captured attention is the Triton kernel (route_sdpa in _capture),
        # which reads fp32 masks and the GPU-scalar seqlen.
        self.use_seqlen = True
        self.mask_dtype = torch.float32
        self.mask_min = torch.finfo(self.mask_dtype).min
        self.cache = StaticCache(config=model.config, max_cache_len=self.max_kv)
        self.seqlen = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.buckets = {}
        # One memory pool for all buckets: replays run one at a time and each
        # bucket's output stays referenced, so sharing is safe and memory does not
        # grow with the number of buckets.
        self.pool = torch.cuda.graph_pool_handle()
        self.param_ptrs = None
        self.prefix_len = 0
        self.n_captures = 0

    # ── per rollout ──────────────────────────────────────────────────────────

    def load_prefix(self, keys, values):
        """Copy the draft prefix KV (list per layer, [1, Hkv, L, D]) into slots [0, L)."""
        ptrs = tuple(p.data_ptr() for p in self.model.parameters())
        if ptrs != self.param_ptrs:
            self.buckets.clear()                     # parameter storage moved → recapture
            self.param_ptrs = ptrs
        L = keys[0].shape[-2]
        assert L <= self.max_kv, (L, self.max_kv)
        for layer, k, v in zip(self.cache.layers, keys, values):
            k, v = k.detach(), v.detach()
            if not layer.is_initialized:
                layer.lazy_initialization(k, v)
            layer.keys[:, :, :L].copy_(k)
            layer.values[:, :, :L].copy_(v)
        self.prefix_len = L

    @torch.no_grad()
    def forward(self, tokens, positions, prefix_allowed, anc_rows):
        """Logits [n, V] for n forest nodes placed right after the cached ones.

        tokens/positions [n]; prefix_allowed [n, L] bool (row sees prefix col);
        anc_rows [n, m] bool over forest slots [0, m), where the n new nodes are
        the last n of them (m = nodes already cached + n).
        """
        n = tokens.numel()
        L = self.prefix_len
        m = anc_rows.shape[1]
        write_start = L + m - n
        q = _bucket(n, self.min_q)
        assert write_start + q <= self.max_kv, "StaticCache too small for this forest"

        b = self.buckets.get(q)
        if b is None:
            b = self.buckets[q] = _Bucket(q, self.max_kv, self.device, self.mask_dtype)
        b.ids.zero_()
        b.ids[0, :n] = tokens
        b.pos.zero_()
        b.pos[0, :n] = positions
        b.mask.fill_(self.mask_min)
        b.mask[0, 0, :n, :L].masked_fill_(prefix_allowed, 0.0)
        b.mask[0, 0, :n, L:L + m].masked_fill_(anc_rows, 0.0)
        self.seqlen.fill_(L + m)
        if self.use_seqlen:
            vis = b.mask[0, 0] > self.mask_min / 2
            rows = vis.shape[0] if b.q < 64 else 64
            b.bmap.copy_(vis.view(-1, rows, vis.shape[1] // 64, 64).any(3).any(1))

        if b.graph is None:
            self._capture(b, write_start)
        self._set_write(write_start)
        b.graph.replay()
        return b.logits[0, :n].clone()

    # ── internals ────────────────────────────────────────────────────────────

    def _set_write(self, start):
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(start)

    def _run(self, b):
        return self.fwd(input_ids=b.ids, attention_mask=b.mask, position_ids=b.pos,
                        past_key_values=self.cache, use_cache=True).logits

    def _capture(self, b, write_start):
        prev = forest_attn._SEQLEN, forest_attn._BMAP
        if self.use_seqlen:
            forest_attn.set_seqlen(self.seqlen, b.bmap)
        else:
            forest_attn.set_seqlen(None)
        route = forest_attn.route_sdpa()
        route.__enter__()
        try:
            # Warm up on a side stream (Triton autotune, lazy allocations), then capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._set_write(write_start)
                    self._run(b)
            torch.cuda.current_stream().wait_stream(s)
            self._set_write(write_start)
            b.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(b.graph, pool=self.pool):
                b.logits = self._run(b)
            self.n_captures += 1
        finally:
            route.__exit__()
            forest_attn.set_seqlen(*prev)
