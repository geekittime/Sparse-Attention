import torch
import flashinfer.decode
import triton
import triton.language as tl

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}
_seq_lens_cache = {}
_lse_partial_cache = {}
_triton_lse_disabled = set()

QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TOPK = 2048
_LOG2E = 1.4426950408889634
_LSE_BLOCK_N = 64
_LSE_NUM_BLOCKS = TOPK // _LSE_BLOCK_N


def _get_workspace(device):
    key = str(device)
    buf = _workspace_cache.get(key)
    if buf is None:
        buf = torch.zeros(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buf
    return buf


def _get_lse_partials(device, num_tokens):
    key = (str(device), num_tokens)
    cached = _lse_partial_cache.get(key)
    if cached is None:
        shape = (num_tokens, 16, _LSE_NUM_BLOCKS)
        partial_m = torch.empty(shape, dtype=torch.float32, device=device)
        partial_d = torch.empty(shape, dtype=torch.float32, device=device)
        cached = (partial_m, partial_d)
        _lse_partial_cache[key] = cached
    return cached


def _get_full_seq_lens(device, num_tokens):
    key = (str(device), num_tokens)
    cached = _seq_lens_cache.get(key)
    if cached is None:
        cached = torch.full((num_tokens,), TOPK, dtype=torch.int32, device=device)
        _seq_lens_cache[key] = cached
    return cached


def _get_seq_lens(sparse_indices):
    num_tokens = sparse_indices.shape[0]
    if bool((sparse_indices[:, -1] != -1).all().item()):
        return _get_full_seq_lens(sparse_indices.device, num_tokens)
    return (sparse_indices != -1).sum(dim=1).to(torch.int32)


def _as_float(scale):
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


@triton.jit
def _lse_stage1_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    sparse_indices,
    partial_m,
    partial_d,
    sm_scale: tl.constexpr,
    TOPK_C: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HAS_INVALID: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)

    offs_h = tl.arange(0, 16)
    offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    raw_idx = tl.load(sparse_indices + token_id * TOPK_C + offs_n)
    valid_n = raw_idx >= 0
    if not HAS_INVALID:
        valid_n = offs_n < TOPK_C
    safe_idx = tl.where(valid_n, raw_idx, 0)

    logits = tl.zeros((16, BLOCK_N), dtype=tl.float32)

    for d_start in tl.static_range(0, D_CKV, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        q = tl.load(
            q_nope + token_id * 16 * D_CKV + offs_h[:, None] * D_CKV + offs_d[None, :]
        )
        k = tl.load(ckv_cache + safe_idx[:, None] * D_CKV + offs_d[None, :])
        logits += tl.dot(q, tl.trans(k), out_dtype=tl.float32)

    offs_pe = tl.arange(0, D_KPE)
    qpe = tl.load(
        q_pe
        + token_id * 16 * D_KPE
        + offs_h[:, None] * D_KPE
        + offs_pe[None, :]
    )
    kpe = tl.load(kpe_cache + safe_idx[:, None] * D_KPE + offs_pe[None, :])
    logits += tl.dot(qpe, tl.trans(kpe), out_dtype=tl.float32)

    logits = logits * (sm_scale * _LOG2E)
    logits = tl.where(valid_n[None, :], logits, -float("inf"))
    m = tl.max(logits, axis=1)
    m_safe = tl.where(m == -float("inf"), 0.0, m)
    d = tl.sum(tl.where(valid_n[None, :], tl.exp2(logits - m_safe[:, None]), 0.0), axis=1)

    out_offsets = (token_id * 16 + offs_h) * NUM_BLOCKS + block_id
    tl.store(partial_m + out_offsets, m)
    tl.store(partial_d + out_offsets, d)


@triton.jit
def _lse_stage2_kernel(
    partial_m,
    partial_d,
    lse,
    BLOCKS: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offs = tl.arange(0, BLOCKS)
    base = (token_id * 16 + head_id) * BLOCKS
    m = tl.load(partial_m + base + offs)
    d = tl.load(partial_d + base + offs)
    m_final = tl.max(m, axis=0)
    m_safe = tl.where(m_final == -float("inf"), 0.0, m_final)
    d_final = tl.sum(d * tl.exp2(m - m_safe), axis=0)
    lse_val = m_final + tl.log2(d_final)
    tl.store(lse + token_id * 16 + head_id, lse_val)


def _compute_lse_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, lse):
    num_tokens = q_nope.shape[0]
    device = q_nope.device
    if str(device) in _triton_lse_disabled:
        return False

    partial_m, partial_d = _get_lse_partials(device, num_tokens)
    _lse_stage1_kernel[(num_tokens, _LSE_NUM_BLOCKS)](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        partial_m,
        partial_d,
        sm_scale,
        TOPK,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        _LSE_NUM_BLOCKS,
        True,
        BLOCK_N=_LSE_BLOCK_N,
        BLOCK_D=64,
        num_warps=4,
        num_stages=3,
    )
    _lse_stage2_kernel[(num_tokens, 16)](
        partial_m,
        partial_d,
        lse,
        BLOCKS=_LSE_NUM_BLOCKS,
        num_warps=1,
        num_stages=1,
    )
    return True


def _compute_reference_outputs(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    sparse_indices,
    sm_scale,
    output,
    lse,
    compute_output,
):
    k_ckv_all = ckv_cache.reshape(-1, KV_LORA_RANK)
    k_pe_all = kpe_cache.reshape(-1, QK_ROPE_HEAD_DIM)
    num_tokens = q_nope.shape[0]
    chunk_size = 16

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        idx = sparse_indices[start:end]
        valid = idx != -1
        safe_idx = torch.where(valid, idx, torch.zeros_like(idx)).to(torch.long)

        k_ckv = k_ckv_all[safe_idx].to(torch.float32)
        k_pe = k_pe_all[safe_idx].to(torch.float32)
        qn = q_nope[start:end].to(torch.float32)
        qp = q_pe[start:end].to(torch.float32)

        logits = torch.einsum("bhd,bkd->bhk", qn, k_ckv)
        logits += torch.einsum("bhd,bkd->bhk", qp, k_pe)
        logits *= sm_scale
        logits = logits.masked_fill(~valid[:, None, :], -float("inf"))

        lse[start:end].copy_(torch.logsumexp(logits, dim=-1) * _LOG2E)

        if compute_output:
            all_invalid = ~valid.any(dim=-1)
            if all_invalid.any():
                logits = logits.clone()
                logits[all_invalid] = 0.0
            attn = torch.softmax(logits, dim=-1)
            out = torch.einsum("bhk,bkd->bhd", attn, k_ckv)
            if all_invalid.any():
                out[all_invalid] = 0.0
            output[start:end].copy_(out.to(output.dtype))


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens = q_nope.shape[0]
    device = q_nope.device

    bmm1_scale = _as_float(sm_scale)

    if num_tokens == 0:
        return None

    query = torch.cat([q_nope, q_pe], dim=-1).unsqueeze(1)  # [T, 1, H, ckv+kpe]
    kv_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)    # [num_pages, page_size, ckv+kpe]
    block_tables = sparse_indices.unsqueeze(1)              # [T, 1, topk]

    # seq_lens = number of valid (non -1) entries per token
    # The kernel only reads the first seq_lens entries from block_tables;
    # valid entries are already contiguous at the front.
    seq_lens = _get_seq_lens(sparse_indices)
    max_seq_len = TOPK
    workspace = _get_workspace(device)

    try:
        flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            out=output,
            sparse_mla_top_k=TOPK,
            bmm1_scale=bmm1_scale,
        )
        compute_output = False
    except TypeError as exc:
        if "sparse_mla_top_k" not in str(exc):
            raise
        compute_output = True
    except ValueError:
        compute_output = True

    if compute_output:
        _compute_reference_outputs(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            sparse_indices,
            bmm1_scale,
            output,
            lse,
            True,
        )
    else:
        try:
            if not _compute_lse_triton(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                sparse_indices,
                bmm1_scale,
                lse,
            ):
                _compute_reference_outputs(
                    q_nope,
                    q_pe,
                    ckv_cache,
                    kpe_cache,
                    sparse_indices,
                    bmm1_scale,
                    output,
                    lse,
                    False,
                )
        except Exception:
            _triton_lse_disabled.add(str(device))
            _compute_reference_outputs(
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                sparse_indices,
                bmm1_scale,
                output,
                lse,
                False,
            )

    return None
