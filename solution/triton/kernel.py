import torch
import flashinfer.decode

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}
_cat_buffer_cache = {}

QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM
TOPK = 2048
_LOG2E = 1.4426950408889634
_SM_SCALE = 0.07216878364870322
_LSE_CHUNK_SIZE = 64


def _get_workspace(device):
    key = str(device)
    buf = _workspace_cache.get(key)
    if buf is None:
        buf = torch.zeros(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buf
    return buf


def _get_cat_buffers(q_nope, ckv_cache):
    num_tokens = q_nope.shape[0]
    num_pages = ckv_cache.shape[0]
    key = (str(q_nope.device), q_nope.dtype, ckv_cache.dtype)
    cached = _cat_buffer_cache.get(key)
    if cached is None:
        query_buf = torch.empty(
            (num_tokens, 16, HEAD_DIM_QK),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        kv_buf = torch.empty(
            (num_pages, 64, HEAD_DIM_QK),
            dtype=ckv_cache.dtype,
            device=ckv_cache.device,
        )
    else:
        query_buf, kv_buf = cached
        if query_buf.shape[0] < num_tokens:
            query_buf = torch.empty(
                (num_tokens, 16, HEAD_DIM_QK),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
        if kv_buf.shape[0] < num_pages:
            kv_buf = torch.empty(
                (num_pages, 64, HEAD_DIM_QK),
                dtype=ckv_cache.dtype,
                device=ckv_cache.device,
            )

    _cat_buffer_cache[key] = (query_buf, kv_buf)
    return query_buf[:num_tokens], kv_buf[:num_pages]


def _compute_lse_fast(
    query_buf,
    kv_cache,
    safe_indices,
    valid_indices,
    sm_scale,
    lse,
):
    kv_all = kv_cache.reshape(-1, HEAD_DIM_QK)
    num_tokens = query_buf.shape[0]

    for start in range(0, num_tokens, _LSE_CHUNK_SIZE):
        end = min(start + _LSE_CHUNK_SIZE, num_tokens)
        safe_idx = safe_indices[start:end]
        valid = valid_indices[start:end]

        k = kv_all[safe_idx]
        q = query_buf[start:end]

        logits = torch.bmm(q, k.transpose(1, 2))
        logits.mul_(sm_scale)
        logits.masked_fill_(~valid[:, None, :], -float("inf"))

        lse[start:end].copy_(torch.logsumexp(logits, dim=-1).to(torch.float32) * _LOG2E)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    num_tokens = q_nope.shape[0]
    device = q_nope.device

    bmm1_scale = _SM_SCALE

    if num_tokens == 0:
        return None

    query_buf, kv_cache = _get_cat_buffers(q_nope, ckv_cache)
    torch.cat([q_nope, q_pe], dim=-1, out=query_buf)
    torch.cat([ckv_cache, kpe_cache], dim=-1, out=kv_cache)
    query = query_buf.unsqueeze(1)                          # [T, 1, H, ckv+kpe]
    block_tables = sparse_indices.unsqueeze(1)              # [T, 1, topk]

    valid_indices = sparse_indices >= 0
    seq_lens = valid_indices.sum(dim=1, dtype=torch.int32)
    safe_indices = sparse_indices.clamp_min(0)
    max_seq_len = TOPK
    workspace = _get_workspace(device)

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
    _compute_lse_fast(
        query_buf,
        kv_cache,
        safe_indices,
        valid_indices,
        bmm1_scale,
        lse,
    )

    return None
