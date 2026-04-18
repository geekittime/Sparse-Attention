import torch
import triton
import triton.language as tl

QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TOPK = 2048
NUM_HEADS = 16
PAGE_SIZE = 64
_LOG2E = 1.4426950408889634

_partial_cache = {}


# ═══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD SPLIT-K  —  one block = all 16 heads × (TOPK/splits) KVs
#  KV loaded ONCE, reused across all heads via tl.dot (tensor cores)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _mh_splitk_kernel(
    q_nope_ptr, q_pe_ptr,
    ckv_ptr, kpe_ptr,
    indices_ptr,
    pa_ptr, pm_ptr, pl_ptr,            # partial accumulators
    sm_scale_log2,
    num_tokens,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_idx_t,
    stride_pa_s, stride_pa_th,
    stride_pm_s, stride_pm_th,
    EPS: tl.constexpr,                 # entries per split
    BK: tl.constexpr,                  # block-K tile
    D: tl.constexpr,                   # KV_LORA_RANK  = 512
    R: tl.constexpr,                   # QK_ROPE_DIM   = 64
    H: tl.constexpr,                   # NUM_HEADS      = 16
):
    token_id = tl.program_id(0)
    split_id = tl.program_id(1)
    if token_id >= num_tokens:
        return

    heads = tl.arange(0, H)            # [16]
    d_nope = tl.arange(0, D)           # [512]
    d_rope = tl.arange(0, R)           # [64]

    # ── Load queries (all heads) ──────────────────────────────────────
    q_nope = tl.load(
        q_nope_ptr + token_id * stride_qn_t
        + heads[:, None] * stride_qn_h + d_nope[None, :]
    )                                                       # [H, D]
    q_pe = tl.load(
        q_pe_ptr + token_id * stride_qp_t
        + heads[:, None] * stride_qp_h + d_rope[None, :]
    )                                                       # [H, R]

    # ── Online-softmax state (per head) ───────────────────────────────
    m_i = tl.full([H], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D], dtype=tl.float32)

    idx_base = token_id * stride_idx_t
    k_begin = split_id * EPS

    for k_off in range(0, EPS, BK):
        k_start = k_begin + k_off
        k_ids = k_start + tl.arange(0, BK)

        indices = tl.load(indices_ptr + idx_base + k_ids)
        valid = indices >= 0
        safe = tl.where(valid, indices, 0)

        # Gathered KV reads (unmasked — invalid idx → row 0; masked via score)
        ckv = tl.load(ckv_ptr + (safe * D)[:, None] + d_nope[None, :])    # [BK, D]
        kpe = tl.load(kpe_ptr + (safe * R)[:, None] + d_rope[None, :])    # [BK, R]

        # Scores via tensor-core matmul: [H, D]@[D, BK] + [H, R]@[R, BK]
        s = tl.dot(q_nope, tl.trans(ckv)).to(tl.float32)                  # [H, BK]
        s += tl.dot(q_pe, tl.trans(kpe)).to(tl.float32)
        s *= sm_scale_log2
        s = tl.where(valid[None, :], s, float('-inf'))

        # Online softmax (exp2-based)
        m_tile = tl.max(s, axis=1)                          # [H]
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.math.exp2(m_i - m_new)                   # [H]
        p = tl.math.exp2(s - m_new[:, None])                # [H, BK]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]
        m_i = m_new

        # Weighted output:  [H, BK] @ [BK, D] → [H, D]
        acc += tl.dot(p.to(ckv.dtype), ckv).to(tl.float32)

    # ── Store partial results ─────────────────────────────────────────
    base_th = token_id * H
    for h in tl.static_range(0, H):
        th = base_th + h
        tl.store(pa_ptr + split_id * stride_pa_s + th * stride_pa_th + d_nope, acc[h, :])
        tl.store(pm_ptr + split_id * stride_pm_s + th * stride_pm_th, m_i[h])
        tl.store(pl_ptr + split_id * stride_pm_s + th * stride_pm_th, l_i[h])


# ═══════════════════════════════════════════════════════════════════════
#  REDUCTION  —  merge split-K partials into final output + LSE
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _reduce_kernel(
    pa_ptr, pm_ptr, pl_ptr,
    out_ptr, lse_ptr,
    num_tokens,
    stride_pa_s, stride_pa_th,
    stride_pm_s, stride_pm_th,
    stride_o_t, stride_o_h,
    stride_lse_t,
    D: tl.constexpr,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    token_id = pid // H
    head_id = pid % H
    if token_id >= num_tokens:
        return

    d = tl.arange(0, D)
    th = token_id * H + head_id

    # First split
    acc = tl.load(pa_ptr + 0 * stride_pa_s + th * stride_pa_th + d)
    m_i = tl.load(pm_ptr + 0 * stride_pm_s + th * stride_pm_th)
    l_i = tl.load(pl_ptr + 0 * stride_pm_s + th * stride_pm_th)

    for s in tl.static_range(1, SPLITS):
        acc_s = tl.load(pa_ptr + s * stride_pa_s + th * stride_pa_th + d)
        m_s = tl.load(pm_ptr + s * stride_pm_s + th * stride_pm_th)
        l_s = tl.load(pl_ptr + s * stride_pm_s + th * stride_pm_th)

        m_new = tl.maximum(m_i, m_s)
        a_old = tl.math.exp2(m_i - m_new)
        a_new = tl.math.exp2(m_s - m_new)
        l_i = l_i * a_old + l_s * a_new
        acc = acc * a_old + acc_s * a_new
        m_i = m_new

    acc /= l_i
    o_off = token_id * stride_o_t + head_id * stride_o_h
    tl.store(out_ptr + o_off + d, acc.to(out_ptr.dtype.element_ty))
    tl.store(lse_ptr + token_id * stride_lse_t + head_id,
             m_i + tl.math.log2(l_i))


# ═══════════════════════════════════════════════════════════════════════
#  SINGLE-PASS MULTI-HEAD  —  no split-K (for large batches)
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _mh_single_kernel(
    q_nope_ptr, q_pe_ptr,
    ckv_ptr, kpe_ptr,
    indices_ptr,
    out_ptr, lse_ptr,
    sm_scale_log2,
    num_tokens,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_idx_t,
    stride_o_t, stride_o_h,
    stride_lse_t,
    BK: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    TOPK_C: tl.constexpr,
):
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return

    heads = tl.arange(0, H)
    d_nope = tl.arange(0, D)
    d_rope = tl.arange(0, R)

    q_nope = tl.load(
        q_nope_ptr + token_id * stride_qn_t
        + heads[:, None] * stride_qn_h + d_nope[None, :]
    )
    q_pe = tl.load(
        q_pe_ptr + token_id * stride_qp_t
        + heads[:, None] * stride_qp_h + d_rope[None, :]
    )

    m_i = tl.full([H], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D], dtype=tl.float32)

    idx_base = token_id * stride_idx_t

    for k_start in range(0, TOPK_C, BK):
        k_ids = k_start + tl.arange(0, BK)
        indices = tl.load(indices_ptr + idx_base + k_ids)
        valid = indices >= 0
        safe = tl.where(valid, indices, 0)

        ckv = tl.load(ckv_ptr + (safe * D)[:, None] + d_nope[None, :])
        kpe = tl.load(kpe_ptr + (safe * R)[:, None] + d_rope[None, :])

        s = tl.dot(q_nope, tl.trans(ckv)).to(tl.float32)
        s += tl.dot(q_pe, tl.trans(kpe)).to(tl.float32)
        s *= sm_scale_log2
        s = tl.where(valid[None, :], s, float('-inf'))

        m_tile = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]
        m_i = m_new
        acc += tl.dot(p.to(ckv.dtype), ckv).to(tl.float32)

    acc /= l_i[:, None]
    o_ptrs = (out_ptr + token_id * stride_o_t
              + heads[:, None] * stride_o_h + d_nope[None, :])
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty))
    tl.store(lse_ptr + token_id * stride_lse_t + heads,
             m_i + tl.math.log2(l_i))


# ═══════════════════════════════════════════════════════════════════════
#  Host helpers
# ═══════════════════════════════════════════════════════════════════════

def _as_float(scale):
    return float(scale.item()) if isinstance(scale, torch.Tensor) else float(scale)


def _get_partials(num_tokens, splits, device):
    key = (str(device), splits)
    th = num_tokens * NUM_HEADS
    cached = _partial_cache.get(key)
    if cached is not None:
        pa, pm, pl = cached
        if pa.shape[1] >= th:
            return pa[:, :th], pm[:, :th], pl[:, :th]
    pa = torch.empty((splits, th, KV_LORA_RANK), dtype=torch.float32, device=device)
    pm = torch.empty((splits, th), dtype=torch.float32, device=device)
    pl = torch.empty((splits, th), dtype=torch.float32, device=device)
    _partial_cache[key] = (pa, pm, pl)
    return pa, pm, pl


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens = q_nope.shape[0]
    if num_tokens == 0:
        return None

    device = q_nope.device
    sm_scale_log2 = _as_float(sm_scale) * _LOG2E

    # Zero-copy flatten (both tensors are contiguous from paged KV cache)
    ckv_flat = ckv_cache.view(-1, KV_LORA_RANK)
    kpe_flat = kpe_cache.view(-1, QK_ROPE_HEAD_DIM)

    BK = 16                            # tile size (≥16 for tensor-core tl.dot)
    NUM_SPLITS = 16                    # split-K parallelism factor
    EPS = TOPK // NUM_SPLITS           # 128 entries per split

    if num_tokens >= 16:
        # ── Large batch: enough blocks from tokens alone ──────────────
        _mh_single_kernel[(num_tokens,)](
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            output, lse, sm_scale_log2, num_tokens,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            sparse_indices.stride(0),
            output.stride(0), output.stride(1),
            lse.stride(0),
            BK=BK, D=KV_LORA_RANK, R=QK_ROPE_HEAD_DIM,
            H=NUM_HEADS, TOPK_C=TOPK,
            num_warps=8, num_stages=2,
        )
    else:
        # ── Small batch: split-K for B200 SM utilisation ──────────────
        pa, pm, pl = _get_partials(num_tokens, NUM_SPLITS, device)

        _mh_splitk_kernel[(num_tokens, NUM_SPLITS)](
            q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
            pa, pm, pl,
            sm_scale_log2, num_tokens,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            sparse_indices.stride(0),
            pa.stride(0), pa.stride(1),
            pm.stride(0), pm.stride(1),
            EPS=EPS, BK=BK, D=KV_LORA_RANK, R=QK_ROPE_HEAD_DIM,
            H=NUM_HEADS,
            num_warps=8, num_stages=2,
        )

        _reduce_kernel[(num_tokens * NUM_HEADS,)](
            pa, pm, pl,
            output, lse, num_tokens,
            pa.stride(0), pa.stride(1),
            pm.stride(0), pm.stride(1),
            output.stride(0), output.stride(1),
            lse.stride(0),
            D=KV_LORA_RANK, H=NUM_HEADS, SPLITS=NUM_SPLITS,
            num_warps=4, num_stages=1,
        )

    return None