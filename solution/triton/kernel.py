import torch
import flashinfer.decode

_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_workspace_cache = {}

QK_NOPE_HEAD_DIM = 128
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
TOPK = 2048
_LOG2E = 1.4426950408889634


def _get_workspace(device):
    key = str(device)
    buf = _workspace_cache.get(key)
    if buf is None:
        buf = torch.zeros(_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device=device)
        _workspace_cache[key] = buf
    return buf


def _as_float(scale):
    if isinstance(scale, torch.Tensor):
        return float(scale.item())
    return float(scale)


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
    seq_lens = (sparse_indices != -1).sum(dim=1).to(torch.int32)
    max_seq_len = int(seq_lens.max().item())
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

    _compute_reference_outputs(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        bmm1_scale,
        output,
        lse,
        compute_output,
    )

    return None
