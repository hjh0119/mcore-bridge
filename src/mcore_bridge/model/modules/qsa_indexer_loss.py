# Copyright (c) ModelScope Contributors. All rights reserved.
"""QSA indexer KL objective (Qwen3.8-Flash-Next tech report §2.1.2).

Teacher: the main attention's own probabilities, detached. For each query row
the per-head softmax is summed over every attention head (an all-reduce under
TP), max-pooled over each complete block of ``R`` keys, restricted to the KL
support and L1-normalized. Student: the indexer's block logits
``sum_h relu(<q_h, k_b>) / sqrt(d)`` -- the same score that routes -- under a
masked log-softmax over the same support. Semantics follow Megatron-LM PR #7234
(``qsa_indexer_kl_loss``):

* dense support (Stage-1 distillation, ``dsa_indexer_use_sparse_loss=False``):
  every visible complete block, and the teacher softmax runs over the whole
  causal prefix of the query's document;
* sparse support (Stage-2, ``dsa_indexer_use_sparse_loss=True``): the detached
  top-k blocks, and the teacher softmax runs over exactly the key set the
  sparse attention attends (selected blocks plus the incomplete tail), so the
  teacher is the distribution the model actually used. The tail is attended but
  never enters the KL.

The objective is reduced like the DSA indexer loss: KL summed per row, then
averaged over valid query rows (rows without a complete block contribute zero
but still count), or summed under ``calculate_per_token_loss``.

Implementation: the rows are processed in bounded-memory chunks and the KL
gradient w.r.t. the index queries / block keys is formed chunk by chunk inside
the forward. Only those two gradients (the size of the indexer outputs) are
kept for backward, never a ``[T, num_blocks]`` matrix, and the teacher is never
recomputed.
"""
import math
import torch
import torch.distributed as dist
from typing import Optional

from .qsa_indexer import QSALossInputs

# Byte budget for one chunk's transient teacher/student workspace. Tests
# monkeypatch it to compare a single chunk against many.
_QSA_KL_CHUNK_BYTES = 512 * 1024 * 1024
# Matches megatron dsa_indexer_loss.INDEXER_LOSS_EPS.
_EPS = 1e-10


class _AttachIndexerGrad(torch.autograd.Function):
    """Return the precomputed loss and replay its precomputed gradients in backward."""

    @staticmethod
    def forward(ctx, q, block_keys, loss, grad_q, grad_block_keys):
        ctx.save_for_backward(grad_q, grad_block_keys)
        return loss.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_q, grad_block_keys = ctx.saved_tensors
        return grad_q * grad_output.to(grad_q.dtype), grad_block_keys * grad_output.to(
            grad_block_keys.dtype), None, None, None


def _normalize(target: torch.Tensor) -> torch.Tensor:
    return target / target.sum(dim=-1, keepdim=True).clamp_min(_EPS)


def _head_summed_probs(query, keys, valid, scale, tp_group):
    """Per-row attention probabilities summed over heads.

    Args:
        query: ``[c, Hq, D]`` rows' main-attention queries.
        keys: ``[c, K, Hkv, D]`` per-row keys or ``[K, Hkv, D]`` shared keys.
        valid: ``[c, K]`` keys each row attends.
    Returns:
        ``[c, K]`` fp32, summed over every attention head (all TP ranks).
    """
    c, hq, d = query.shape
    hkv = keys.shape[-2]
    q = query.float().view(c, hkv, hq // hkv, d)
    if keys.dim() == 4:
        scores = torch.einsum('cngd,cknd->cngk', q, keys.float())
    else:
        scores = torch.einsum('cngd,knd->cngk', q, keys.float())
    scores.mul_(scale)
    live = valid.any(dim=-1)
    scores.masked_fill_(~valid[:, None, None, :], float('-inf'))
    # rows without any attended key would softmax to NaN; they carry no teacher mass
    scores.masked_fill_(~live[:, None, None, None], 0.)
    probs = torch.softmax(scores, dim=-1).sum(dim=(1, 2))
    probs = probs.masked_fill(~valid, 0.)
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(probs, group=tp_group)
    return probs


def _student_kl(q_rows, block_keys, target, support, row_weight):
    """``sum_rows row_weight * KL(target || softmax(logits|support))`` (differentiable)."""
    if block_keys.dim() == 3:
        scores = torch.einsum('chd,ckd->chk', q_rows, block_keys)
    else:
        scores = torch.einsum('chd,kd->chk', q_rows, block_keys)
    logits = torch.relu(scores).sum(dim=1) / math.sqrt(q_rows.shape[-1])
    has_support = support.any(dim=-1, keepdim=True)
    logits = logits.masked_fill(~support, float('-inf'))
    logits = torch.where(has_support, logits, torch.zeros_like(logits))
    log_probs = torch.log_softmax(logits, dim=-1).masked_fill(~support, 0.)
    terms = target * (torch.log(target.clamp_min(_EPS)) - log_probs)
    terms = terms.masked_fill(~support, 0.)
    return (terms.sum(dim=-1) * row_weight).sum()


def qsa_indexer_kl_loss(inputs: QSALossInputs,
                        query: torch.Tensor,
                        key: torch.Tensor,
                        *,
                        softmax_scale: float,
                        compress_ratio: int,
                        loss_coeff: float,
                        query_rows: Optional[torch.Tensor] = None,
                        valid_rows: Optional[torch.Tensor] = None,
                        tp_group=None,
                        calculate_per_token_loss: bool = False) -> torch.Tensor:
    """Scaled QSA indexer KL; gradient reaches only ``inputs.q`` / ``inputs.block_keys``.

    Args:
        inputs: Indexer outputs over the full pack space (``QSALossInputs``);
            ``inputs.top_blocks is None`` selects the dense support.
        query: Main-attention queries ``[Tq, Hq_local, D]`` (post-RoPE) of the
            rows in ``query_rows``; detached here.
        key: Main-attention keys ``[T, Hkv_local, D]`` (post-RoPE) of the whole
            pack space; detached here.
        softmax_scale: The main attention's softmax scale.
        compress_ratio: Block size ``R``.
        loss_coeff: Objective multiplier.
        query_rows: Pack-space row of each ``query`` row (CP keeps its local
            rows only); ``None`` means all ``T`` rows in order.
        valid_rows: ``[T]`` bool, ``False`` for padding rows (excluded from both
            the sum and the mean's denominator).
        tp_group: Group sharding the attention heads; the teacher is summed over it.
        calculate_per_token_loss: Return the row sum instead of the row mean.

    Returns:
        Scalar loss.
    """
    R = compress_ratio
    q_all = inputs.q.detach()
    bk_all = inputs.block_keys.detach()
    device = q_all.device
    T = q_all.shape[0]
    query = query.detach()
    key = key.detach()
    rows = torch.arange(T, device=device) if query_rows is None else query_rows.to(device=device, dtype=torch.long)
    if rows.numel() != query.shape[0]:
        raise ValueError(f'QSA indexer KL got {query.shape[0]} query rows for {rows.numel()} row ids.')
    if key.shape[0] != T:
        raise ValueError(f'QSA indexer KL needs keys over the whole pack space ({T}), got {key.shape[0]}.')

    cu = inputs.cu_seqlens.to(device=device, dtype=torch.long)
    doc_lens = cu[1:] - cu[:-1]
    num_docs = doc_lens.numel()
    full_blocks = doc_lens // R
    block_offset = torch.cumsum(full_blocks, 0) - full_blocks
    token_doc = torch.repeat_interleave(torch.arange(num_docs, device=device), doc_lens)
    row_weight = torch.ones(rows.numel(), dtype=torch.float32, device=device)
    if valid_rows is not None:
        row_weight = valid_rows.to(device)[rows].float()
    if calculate_per_token_loss:
        scale = loss_coeff
    else:
        scale = loss_coeff / max(float(row_weight.sum().item()), 1.0)

    grad_q = torch.zeros(q_all.shape, dtype=torch.float32, device=device)
    grad_bk = torch.zeros(bk_all.shape, dtype=torch.float32, device=device)
    bk_leaf = bk_all.float().requires_grad_(True)
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    hq, d = query.shape[1], query.shape[2]
    hkv = key.shape[1]
    budget = _QSA_KL_CHUNK_BYTES

    def accumulate(local, target, support, student_keys):
        nonlocal loss_sum
        with torch.enable_grad():
            q_rows = q_all[rows[local]].float().requires_grad_(True)
            chunk_loss = _student_kl(q_rows, student_keys(bk_leaf), target, support, row_weight[local]) * scale
            gq, gk = torch.autograd.grad(chunk_loss, (q_rows, bk_leaf))
        grad_q[rows[local]] = gq
        grad_bk.add_(gk)
        loss_sum = loss_sum + chunk_loss.detach()

    with torch.no_grad():
        if inputs.top_blocks is None:
            # ---- dense support: per document, the whole causal prefix ----
            row_doc = token_doc[rows]
            order = torch.argsort(row_doc, stable=True)
            counts = torch.bincount(row_doc, minlength=num_docs).tolist()
            start = 0
            for doc, count in enumerate(counts):
                if count == 0:
                    continue
                doc_local = order[start:start + count]
                start += count
                a, length, nb = int(cu[doc]), int(doc_lens[doc]), int(full_blocks[doc])
                off = int(block_offset[doc])
                doc_keys = key[a:a + length]
                key_pos = torch.arange(length, device=device)
                chunk = max(1, budget // max(1, hq * length * 4 * 3))
                for c0 in range(0, count, chunk):
                    local = doc_local[c0:c0 + chunk]
                    pos = rows[local] - a
                    probs = _head_summed_probs(query[local], doc_keys, key_pos[None, :] <= pos[:, None], softmax_scale,
                                               tp_group)
                    support = torch.arange(nb, device=device)[None, :] < ((pos + 1) // R)[:, None]
                    pooled = probs[:, :nb * R].reshape(local.numel(), nb, R).amax(dim=-1)
                    target = _normalize(pooled.masked_fill(~support, 0.))
                    accumulate(local, target, support, lambda bk, o=off, n=nb: bk[o:o + n])
        else:
            # ---- sparse support: the detached top-k blocks + the attended tail ----
            top_blocks = inputs.top_blocks
            keep = inputs.keep
            k = top_blocks.shape[1]
            block_doc = torch.repeat_interleave(torch.arange(num_docs, device=device), full_blocks)
            block_first = cu[block_doc] + (torch.arange(block_doc.numel(), device=device) - block_offset[block_doc]) * R
            arange_r = torch.arange(R, device=device)
            width = k * R + R
            per_row = width * hkv * d * key.element_size() + 3 * hq * width * 4 + 2 * k * q_all.shape[-1] * 4
            chunk = max(1, budget // max(1, per_row))
            for c0 in range(0, rows.numel(), chunk):
                local = torch.arange(c0, min(c0 + chunk, rows.numel()), device=device)
                r = rows[local]
                tb = top_blocks[r]
                kp = keep[r]
                block_tok = (block_first[tb].unsqueeze(-1) + arange_r).flatten(1)
                doc_start = cu[token_doc[r]]
                tail = (doc_start + (r - doc_start + 1) // R * R).unsqueeze(-1) + arange_r
                idx = torch.cat([block_tok, tail], dim=1)
                attended = torch.cat([kp.repeat_interleave(R, dim=1), tail <= r.unsqueeze(-1)], dim=1)
                # masked slots may point past the pack end (the last rows' tail); clamp, they stay masked
                probs = _head_summed_probs(query[local], key[idx.clamp(0, T - 1)], attended, softmax_scale, tp_group)
                pooled = probs[:, :k * R].reshape(local.numel(), k, R).amax(dim=-1)
                target = _normalize(pooled.masked_fill(~kp, 0.))
                accumulate(local, target, kp, lambda bk, t=tb: bk[t])

    return _AttachIndexerGrad.apply(inputs.q, inputs.block_keys, loss_sum, grad_q.to(inputs.q.dtype),
                                    grad_bk.to(inputs.block_keys.dtype))
