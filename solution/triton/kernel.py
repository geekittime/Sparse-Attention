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


@triton.jit
def _mh_splitk_kernel(
    q_nope_ptr, q_pe_ptr,
    ckv_ptr, kpe_ptr,
    indices_ptr,
    pa_ptr, pm_ptr, pl_ptr,
    sm_scale_log2,
    num_tokens,
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_idx_t,
    stride_pa_s, stride_pa_th,
    stride_pm_s, stride_pm_th,
    EPS: tl.constexpr,
    BK: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
):
    token_id = tl.program_id(0)
    split_id = tl.program_id(1)
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

    m_i = tl.full([H], -3.4028234663852886e38, dtype=tl.float32)
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

        ckv = tl.load(ckv_ptr + (safe * D)[:, None] + d_nope[None, :])
        kpe = tl.load(kpe_ptr + (safe * R)[:, None] + d_rope[None, :])

        s = tl.dot(q_nope, tl.trans(ckv)).to(tl.float32)
        s += tl.dot(q_pe, tl.trans(kpe)).to(tl.float32)
        s *= sm_scale_log2
        s = tl.where(valid[None, :], s, -3.4028234663852886e38)

        m_tile = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]
        m_i = m_new

        acc += tl.dot(p.to(ckv.dtype), ckv).to(tl.float32)

    base_th = token_id * H
    th = base_th + heads
    pa_offsets = (
        pa_ptr
        + split_id * stride_pa_s
        + th[:, None] * stride_pa_th
        + d_nope[None, :]
    )
    stat_offsets = split_id * stride_pm_s + th * stride_pm_th
    tl.store(pa_offsets, acc)
    tl.store(pm_ptr + stat_offsets, m_i)
    tl.store(pl_ptr + stat_offsets, l_i)


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

    has_data = l_i > 0.0
    denom = tl.where(has_data, l_i, 1.0)
    acc /= denom
    o_off = token_id * stride_o_t + head_id * stride_o_h
    tl.store(out_ptr + o_off + d, acc.to(out_ptr.dtype.element_ty))
    tl.store(
        lse_ptr + token_id * stride_lse_t + head_id,
        tl.where(has_data, m_i + tl.math.log2(l_i), float('-inf')),
    )


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

    m_i = tl.full([H], -3.4028234663852886e38, dtype=tl.float32)
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
        s = tl.where(valid[None, :], s, -3.4028234663852886e38)

        m_tile = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        p = tl.where(valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]
        m_i = m_new
        acc += tl.dot(p.to(ckv.dtype), ckv).to(tl.float32)

    has_data = l_i > 0.0
    denom = tl.where(has_data, l_i, 1.0)
    acc /= denom[:, None]
    o_ptrs = (
        out_ptr + token_id * stride_o_t
        + heads[:, None] * stride_o_h + d_nope[None, :]
    )
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty))
    tl.store(
        lse_ptr + token_id * stride_lse_t + heads,
        tl.where(has_data, m_i + tl.math.log2(l_i), float('-inf')),
    )


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


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens = q_nope.shape[0]
    if num_tokens == 0:
        return None

    device = q_nope.device
    sm_scale_log2 = _as_float(sm_scale) * _LOG2E

    ckv_flat = ckv_cache.view(-1, KV_LORA_RANK)
    kpe_flat = kpe_cache.view(-1, QK_ROPE_HEAD_DIM)

    BK = 16
    NUM_SPLITS = 16
    EPS = TOPK // NUM_SPLITS

    if num_tokens >= 16:
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
