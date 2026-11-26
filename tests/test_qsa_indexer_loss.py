# Copyright (c) ModelScope Contributors. All rights reserved.
"""QSA indexer KL (``qsa_indexer_kl_loss``) against the Megatron-LM PR #7234 reference.

The reference below is the materialized PyTorch oracle of NVIDIA/Megatron-LM#7234
(``qsa_indexer_loss.py`` / ``qsa.py``), restated on the released DSA helpers: token
teacher = head-summed attention probabilities, max-pooled onto complete blocks,
restricted to the support and L1-normalized; student = ``sum_h relu(q.k) / sqrt(d)``
under a masked log-softmax; dense support = every visible complete block, sparse
support = the canonical (score desc, block id asc) top-k. The attention the teacher
is taken from is the one QSA runs: full causal for the dense support, the selected
blocks plus the incomplete tail for the sparse one.
"""
import copy
import math
import pytest
import torch
from megatron.core.transformer.experimental_attention_variant import dsa_indexer_loss, dsa_masking

import mcore_bridge.model.modules.qsa_indexer as qi
import mcore_bridge.model.modules.qsa_indexer_loss as ql
from mcore_bridge.model.modules.qsa_indexer import QSAIndexer
from mcore_bridge.model.modules.qsa_indexer_loss import qsa_indexer_kl_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='QSA indexer KL tests build TELinear on GPU')

R, BUDGET = 4, 32  # block_topk = 8
HQ, HKV, DM = 4, 2, 32  # main attention: GQA with 2 query heads per kv head
SCALE = DM**-0.5


def _config(coeff=1.0, sparse=True):
    from test_qwen4_exp_units import _make_config
    cfg = _make_config(compress_ratio=R, budget=BUDGET)
    cfg.dsa_indexer_loss_coeff = coeff
    cfg.dsa_indexer_use_sparse_loss = sparse
    return cfg


def _indexer(coeff=1.0, sparse=True, seed=0):
    torch.manual_seed(seed)
    idx = QSAIndexer(_config(coeff, sparse)).cuda()
    with torch.no_grad():
        idx.index_qk_proj.weight.normal_(0, 0.05)
        idx.q_layernorm.weight.normal_(0, 0.1)
        idx.k_layernorm.weight.normal_(0, 0.1)
    return idx


def _inputs(s, b, seed, h=256, rot=32):
    g = torch.Generator(device='cuda').manual_seed(seed)
    hidden = torch.randn(s, b, h, device='cuda', generator=g)
    freqs = torch.rand(s, 1, 1, rot, device='cuda', generator=g) * s  # partial rope: rot < index head dim
    q = torch.randn(s, b, HQ, DM, device='cuda', generator=g) * 2
    k = torch.randn(s, b, HKV, DM, device='cuda', generator=g) * 2
    return hidden, freqs, q, k


def _flat(x):  # [s, b, H, D] -> batch-major [b * s, H, D], the indexer's pack space
    return x.transpose(0, 1).reshape(-1, *x.shape[2:])


def _grads(module):
    return {n: p.grad.detach().clone() for n, p in module.named_parameters()}


def _assert_grads_close(actual, expected, rel=1e-5):
    """fp32 parity up to accumulation order: the gradients reach the weights through the same
    TELinear wgrad in both runs, but the row chunks sum in a different order."""
    for name, ref in expected.items():
        got = actual[name]
        err = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        assert err <= rel, f'{name}: relative L2 error {err:.3e} > {rel:.0e}'
        torch.testing.assert_close(
            got, ref, rtol=0, atol=1e-3 * ref.abs().amax().item(), msg=lambda m, n=name: f'{n}: {m}')


# ---------------------------------------------------------------- reference (PR #7234)
def _ref_block_logits(q_idx, keys):
    return torch.relu(torch.einsum('bqhd,bpd->bqhp', q_idx.float(), keys.float())).sum(dim=2) / math.sqrt(
        q_idx.shape[-1])


def _ref_causal(s, b, nb):
    visible = (torch.arange(s, device='cuda') + 1) // R
    return (torch.arange(nb, device='cuda')[None, :] < visible[:, None])[None].expand(b, s, nb)


def _ref_topk_support(block_logits, causal, budget):
    masked = block_logits.detach().masked_fill(~causal, -torch.inf)
    width = min(budget, block_logits.size(-1))
    chosen = torch.zeros_like(causal)
    chosen.scatter_(-1, torch.sort(masked, dim=-1, descending=True, stable=True).indices[..., :width], True)
    visible = causal.sum(dim=-1, keepdim=True)
    return torch.where(visible > budget, causal & chosen, causal)


def _ref_teacher(q, k, allowed):
    """``allowed`` [b, s, s] -> detached head-summed L1-normalized token teacher [b, s, s]."""
    qh = q.permute(1, 2, 0, 3).float()  # [b, HQ, s, D]
    kh = k.permute(1, 2, 0, 3).float().repeat_interleave(HQ // HKV, dim=1)
    scores = qh @ kh.transpose(-1, -2) * SCALE
    probs = dsa_masking.masked_softmax(scores, allowed[:, None].expand_as(scores))
    return dsa_indexer_loss.normalize_indexer_target(probs.sum(dim=1))


def _ref_kl(block_logits, teacher_tok, support, coeff, pool='max', per_token=False, query_valid_rows=None):
    nb = block_logits.size(-1)
    blocks = teacher_tok[..., :nb * R].unflatten(-1, (nb, R))
    pooled = blocks.amax(dim=-1) if pool == 'max' else blocks.mean(dim=-1)
    target = dsa_indexer_loss.normalize_indexer_target(pooled.masked_fill(~support, 0.0))
    log_probs = dsa_masking.masked_log_softmax(block_logits, support)
    return dsa_indexer_loss.indexer_loss_from_target(
        target,
        log_probs,
        coeff,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=per_token,
        valid_mask=support)


def _allowed_from_indices(indices, s):
    b = indices.shape[0]
    allowed = torch.zeros(b, s, s + 1, dtype=torch.bool, device='cuda')
    allowed.scatter_(-1, torch.where(indices >= 0, indices, s), True)
    return allowed[..., :s]


def _reference(idx, hidden, freqs, q, k, sparse, coeff, pool='max', per_token=False, projected=None):
    s, b, _ = hidden.shape
    nb = s // R
    q_idx, keys = idx._project_sbhd(hidden, freqs) if projected is None else projected
    block_logits = _ref_block_logits(q_idx, keys)
    causal = _ref_causal(s, b, nb)
    causal_tok = torch.ones(s, s, dtype=torch.bool, device='cuda').tril()[None].expand(b, s, s)
    if sparse:
        support = _ref_topk_support(block_logits, causal, BUDGET // R)
        indices = idx.selection_as_token_indices(hidden, freqs)
        allowed = causal_tok if indices is None else _allowed_from_indices(indices, s)
    else:
        support, allowed = causal, causal_tok
    teacher = _ref_teacher(q, k, allowed)
    return _ref_kl(block_logits, teacher, support, coeff, pool=pool, per_token=per_token), support


def _ours(idx, hidden, freqs, q, k, sparse, coeff, per_token=False):
    if sparse:
        _, inputs = idx.selection_as_token_indices(hidden, freqs, return_loss_inputs=True)
    else:
        inputs = idx.project_loss_inputs_sbhd(hidden, freqs)
    loss = qsa_indexer_kl_loss(
        inputs,
        _flat(q),
        _flat(k),
        softmax_scale=SCALE,
        compress_ratio=R,
        loss_coeff=coeff,
        calculate_per_token_loss=per_token)
    return loss, inputs


def _support_mask(inputs, num_blocks):
    if inputs.top_blocks is None:
        return None
    mask = torch.zeros(inputs.top_blocks.shape[0], num_blocks + 1, dtype=torch.bool, device='cuda')
    mask.scatter_(1, torch.where(inputs.keep, inputs.top_blocks, num_blocks), True)
    return mask[:, :num_blocks]


# ---------------------------------------------------------------- tests
@pytest.mark.parametrize('sparse', [True, False])
@pytest.mark.parametrize('s', [3, 30, 75, 130])  # no block / below budget (no-op) / sparse / sparse + tail
@pytest.mark.parametrize('per_token', [False, True])
def test_kl_matches_upstream_reference(sparse, s, per_token):
    b, coeff = 2, 0.7
    nb = s // R
    idx = _indexer(sparse=sparse)
    ref_idx = copy.deepcopy(idx)
    hidden, freqs, q, k = _inputs(s, b, seed=s)

    loss, inputs = _ours(idx, hidden, freqs, q, k, sparse, coeff, per_token)
    grad_q, grad_k = torch.autograd.grad(loss, (inputs.q, inputs.block_keys), retain_graph=True)
    loss.backward()
    # 1) the KL itself, on the very same index q / block keys (fp32, no GEMM in between)
    leaf_q = inputs.q.detach().view(b, s, *inputs.q.shape[1:]).requires_grad_(True)
    leaf_k = inputs.block_keys.detach().view(b, nb, inputs.block_keys.shape[-1]).requires_grad_(True)
    ref_loss, ref_support = _reference(
        idx, hidden, freqs, q, k, sparse, coeff, per_token=per_token, projected=(leaf_q, leaf_k))
    ref_gq, ref_gk = torch.autograd.grad(ref_loss, (leaf_q, leaf_k))
    assert torch.isfinite(loss).item()
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-7)
    for got, ref in ((grad_q, ref_gq), (grad_k, ref_gk)):
        ref = ref.reshape(got.shape)
        torch.testing.assert_close(
            got, ref, rtol=1e-4, atol=1e-5 * ref.abs().amax().item() + 1e-12 if ref.numel() else 0)
    # 2) end to end into the indexer weights: TE's fp32 wgrad runs on TF32, so near-equal
    # upstream grads may round apart in single elements -- compare in norm
    ref_full, _ = _reference(ref_idx, hidden, freqs, q, k, sparse, coeff, per_token=per_token)
    ref_full.backward()
    _assert_grads_close(_grads(idx), _grads(ref_idx), rel=2e-4)
    # the sparse support is exactly the canonical top-k (block-local ids, batch-major rows)
    support = _support_mask(inputs, b * nb)
    if support is not None:
        local = support.view(b, s, b, nb)[torch.arange(b), :, torch.arange(b)]
        assert torch.equal(local, ref_support)
    if s >= 2 * R:
        assert loss.item() > 0


def test_reference_is_sensitive_to_max_pooling():
    """Negative control: a mean-pooled teacher must not pass the parity above."""
    idx = _indexer()
    hidden, freqs, q, k = _inputs(130, 2, seed=1)
    loss, _ = _ours(idx, hidden, freqs, q, k, True, 1.0)
    mean_loss, _ = _reference(idx, hidden, freqs, q, k, True, 1.0, pool='mean')
    assert not torch.allclose(loss, mean_loss, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize('sparse', [True, False])
@pytest.mark.parametrize('ties', [False, True])
def test_chunking_and_ties_do_not_change_routes_or_loss(monkeypatch, sparse, ties):
    idx = _indexer(sparse=sparse)
    hidden, freqs, q, k = _inputs(130, 2, seed=3)
    if ties:
        hidden.zero_()  # every block scores exactly 0 -> the tie-break decides the whole route
    results = []
    for kl_bytes, score_bytes in [(1 << 40, 1 << 40), (1, 1024)]:
        monkeypatch.setattr(ql, '_QSA_KL_CHUNK_BYTES', kl_bytes)
        monkeypatch.setattr(qi, '_QSA_INDEX_SCORE_CHUNK_BYTES', score_bytes)
        idx.zero_grad(set_to_none=True)
        indices = idx.selection_as_token_indices(hidden, freqs)
        loss, inputs = _ours(idx, hidden, freqs, q, k, sparse, 1.0)
        loss.backward()
        results.append((indices, inputs.top_blocks, inputs.keep, loss.detach(), _grads(idx)))
    (i0, t0, k0, l0, g0), (i1, t1, k1, l1, g1) = results  # noqa
    assert torch.equal(i0, i1)
    if sparse:
        assert torch.equal(t0, t1) and torch.equal(k0, k1)
    assert torch.isfinite(l0).item()
    torch.testing.assert_close(l0, l1, rtol=1e-6, atol=1e-7)
    _assert_grads_close(g1, g0, rel=2e-4)


@pytest.mark.parametrize('encoding', ['mask', 'indices', 'thd'])
def test_trainable_projection_routes_bitwise_like_frozen(encoding):
    """The loss path projects once with grad; its routing must equal the frozen no-grad path."""
    idx = _indexer()
    hidden, freqs, _, _ = _inputs(130, 2, seed=5)
    if encoding == 'mask':
        plain = idx.selection_as_mask(hidden, freqs)
        routed, _ = idx.selection_as_mask(hidden, freqs, return_loss_inputs=True)
    elif encoding == 'indices':
        plain = idx.selection_as_token_indices(hidden, freqs)
        routed, _ = idx.selection_as_token_indices(hidden, freqs, return_loss_inputs=True)
    else:
        cu = torch.tensor([0, 70, 130, 260], device='cuda')
        tok = hidden.transpose(0, 1).reshape(260, -1)
        f = torch.cat([freqs[:70], freqs[:60], freqs[:130]]).reshape(260, 1, 1, -1)
        plain = idx.select_token_indices_thd(tok, f, cu)
        routed, _ = idx.select_token_indices_thd(tok, f, cu, return_loss_inputs=True)
    assert plain is not None and torch.equal(plain, routed)


def test_gradient_reaches_only_the_indexer():
    idx = _indexer()
    hidden, freqs, q, k = _inputs(130, 2, seed=7)
    hidden.requires_grad_(True)
    q.requires_grad_(True)
    k.requires_grad_(True)
    loss, inputs = _ours(idx, hidden, freqs, q, k, True, 1.0)
    assert inputs.q.requires_grad and inputs.block_keys.requires_grad
    assert not inputs.top_blocks.requires_grad
    loss.backward()
    assert hidden.grad is None and q.grad is None and k.grad is None
    for name, p in idx.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_trainable_flag_and_tp_grad_averaging():
    frozen = _indexer(coeff=0.0)
    assert all(not p.requires_grad for p in frozen.parameters())
    assert not any(hasattr(p, 'average_gradients_across_tp_domain') for p in frozen.parameters())
    trainable = _indexer(coeff=0.1)
    # finalize_model_grads otherwise SUMs SP params and any `q_layernorm`/`k_layernorm` across TP
    assert all(p.requires_grad and p.average_gradients_across_tp_domain for p in trainable.parameters())


def _packed(lens, seed):
    g = torch.Generator(device='cuda').manual_seed(seed)
    T = sum(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device='cuda')
    pos = torch.cat([torch.arange(n) for n in lens]).cuda()
    base = torch.rand(max(lens), 1, 1, 32, device='cuda', generator=g) * 100
    hidden = torch.randn(T, 256, device='cuda', generator=g)
    q = torch.randn(T, HQ, DM, device='cuda', generator=g) * 2
    k = torch.randn(T, HKV, DM, device='cuda', generator=g) * 2
    return cu, base[pos], base, hidden, q, k


@pytest.mark.parametrize('sparse', [True, False])
def test_thd_equals_sum_of_per_document_sbhd(sparse):
    """Packed docs (shorter than a block / within budget / sparse) never mix teacher or support."""
    lens = [3, 20, 75, 130]
    cu, freqs, base, hidden, q, k = _packed(lens, seed=11)
    idx = _indexer(sparse=sparse)
    ref_idx = copy.deepcopy(idx)
    if sparse:
        _, inputs = idx.select_token_indices_thd(hidden, freqs, cu, return_loss_inputs=True)
    else:
        inputs = idx.project_loss_inputs_thd(hidden, freqs, cu)
    loss = qsa_indexer_kl_loss(
        inputs, q, k, softmax_scale=SCALE, compress_ratio=R, loss_coeff=1.0, calculate_per_token_loss=True)
    loss.backward()

    ref = 0.
    for a, n in zip(cu[:-1].tolist(), lens):
        doc = slice(a, a + n)
        doc_loss, _ = _ours(
            ref_idx,
            hidden[doc].unsqueeze(1),
            base[:n],
            q[doc].unsqueeze(1),
            k[doc].unsqueeze(1),
            sparse,
            1.0,
            per_token=True)
        doc_loss.backward()
        ref = ref + doc_loss.detach()
    torch.testing.assert_close(loss.detach(), ref, rtol=1e-5, atol=1e-6)
    _assert_grads_close(_grads(idx), _grads(ref_idx), rel=2e-4)


def test_padding_rows_and_query_subset():
    """``valid_rows`` drop rows from sum and denominator; ``query_rows`` (CP shard) score just those rows."""
    lens = [75, 130]
    cu, freqs, _, hidden, q, k = _packed(lens, seed=13)
    idx = _indexer()
    _, inputs = idx.select_token_indices_thd(hidden, freqs, cu, return_loss_inputs=True)
    T = q.shape[0]
    valid = torch.ones(T, dtype=torch.bool, device='cuda')
    valid[70:75] = False  # padding at the end of doc 0
    valid[200:] = False  # and of doc 1
    kwargs = dict(softmax_scale=SCALE, compress_ratio=R, loss_coeff=1.0)
    with torch.no_grad():
        mean = qsa_indexer_kl_loss(inputs, q, k, valid_rows=valid, **kwargs)
        total = qsa_indexer_kl_loss(inputs, q, k, valid_rows=valid, calculate_per_token_loss=True, **kwargs)
        rows = valid.nonzero().squeeze(1)
        subset = qsa_indexer_kl_loss(inputs, q[rows], k, query_rows=rows, calculate_per_token_loss=True, **kwargs)
        everything = qsa_indexer_kl_loss(inputs, q, k, calculate_per_token_loss=True, **kwargs)
    torch.testing.assert_close(mean * valid.sum(), total, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(subset, total, rtol=1e-6, atol=1e-7)
    assert everything > total  # the padding rows did carry KL, so excluding them matters


def test_rows_without_complete_block_count_but_add_nothing():
    """Upstream semantics: rows before the first complete block contribute 0 yet enter the mean."""
    idx = _indexer(sparse=False)
    hidden, freqs, q, k = _inputs(9, 1, seed=17)  # rows 0..2 see no complete block
    inputs = idx.project_loss_inputs_sbhd(hidden, freqs)
    kwargs = dict(softmax_scale=SCALE, compress_ratio=R, loss_coeff=1.0)
    with torch.no_grad():
        mean = qsa_indexer_kl_loss(inputs, _flat(q), _flat(k), **kwargs)
        total = qsa_indexer_kl_loss(inputs, _flat(q), _flat(k), calculate_per_token_loss=True, **kwargs)
        rows = torch.arange(3, device='cuda')
        head = qsa_indexer_kl_loss(
            inputs, _flat(q)[rows], _flat(k), query_rows=rows, calculate_per_token_loss=True, **kwargs)
    torch.testing.assert_close(mean * 9, total)
    assert head.item() == 0.0
