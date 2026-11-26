# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end QSA indexer KL on a tiny Qwen3.8-Flash-Next (mcore ``qwen4_exp``) model.

Drives the real ``Qwen4ExpLayer`` path (indexer projection -> routing -> main attention
-> teacher capture -> KL -> ``DSAIndexerLossAutoScaler``) for the decoder QSA layer and
the MTP inner layer, and checks the loss against the Megatron-LM PR #7234 reference
recomputed from the attention inputs observed by an independent forward pre-hook.
"""
import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import (DSAIndexerLossAutoScaler,
                                                                          DSAIndexerLossLoggingHelper)
from test_qsa_indexer_loss import _ref_block_logits, _ref_causal, _ref_kl, _ref_topk_support
from test_qwen4_exp_mtp import _lm, _mcore_config, _parallel_context, _tiny_qwen4exp_config

import mcore_bridge.model.gpts.qwen4_exp as qwen4_exp
from mcore_bridge.model.register import get_mcore_model

BUDGET, R = 32, 4  # block_topk = 8, so a 96-token sequence (24 blocks) is genuinely sparse


@pytest.fixture(autouse=True)
def _clean_indexer_state(monkeypatch):
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, 'tracker', {})
    DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    yield
    DSAIndexerLossAutoScaler.main_loss_backward_scale = None


def _build(coeff=0.5, sparse=True, budget=BUDGET, mtp=1, seed=5, **overrides):
    # parameters initialize on the model-parallel CUDA RNG; reseed it so every build is identical
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    hf_config = _tiny_qwen4exp_config()
    hf_config.text_config.indexer_budget = budget
    config = _mcore_config(hf_config, mtp=mtp)
    config.dsa_indexer_loss_coeff = coeff
    config.dsa_indexer_use_sparse_loss = sparse
    for key, value in overrides.items():
        setattr(config, key, value)
    model = get_mcore_model(config)[0].cuda()
    return _lm(model).train(), config


def _batch(b=2, s=96, seed=23):
    g = torch.Generator(device='cuda').manual_seed(seed)
    ids = torch.randint(1, 500, (b, s), device='cuda', generator=g)
    position_ids = torch.arange(s, device='cuda').unsqueeze(0).expand(b, -1)
    mask = torch.triu(torch.ones(b, 1, s, s, device='cuda', dtype=torch.bool), diagonal=1)
    labels = torch.randint(1, 500, (b, s), device='cuda', generator=g)
    loss_mask = torch.ones(b, s, device='cuda', dtype=torch.bool)
    return ids, position_ids, mask, labels, loss_mask


def _run(lm, batch, packed_seq_params=None):
    ids, position_ids, mask, labels, loss_mask = batch
    out = lm(ids, position_ids, mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params)
    loss = out.float().mean()
    loss.backward()
    return loss.detach()


def _qsa_layers(lm):
    layers = [layer for layer in lm.decoder.layers if getattr(layer.self_attention, 'indexer', None) is not None]
    if getattr(lm, 'mtp', None) is not None:
        layers += [layer.mtp_model_layer for layer in lm.mtp.layers]
    return layers


def _grads(lm, indexer):
    return {
        name: p.grad.detach().float().clone()
        for name, p in lm.named_parameters() if ('indexer' in name) == indexer and p.grad is not None
    }


def _tracker():
    values = DSAIndexerLossLoggingHelper.tracker.get('values')
    return None if values is None else values.clone()


@pytest.mark.parametrize('sparse', [True, False])
def test_layer_kl_matches_reference_from_observed_attention(monkeypatch, sparse):
    """The KL each QSA layer attaches equals the PR #7234 reference built from the q/k/mask that
    actually entered core attention (seen by an independent pre-hook), for the decoder and MTP."""
    with _parallel_context():
        lm, _ = _build(sparse=sparse)
        observed, recorded = {}, {}
        for layer in _qsa_layers(lm):

            def hook(module, args, kwargs, layer=layer):
                observed[id(layer)] = (args[0].detach(), args[1].detach(), args[3])

            layer.self_attention.core_attention.register_forward_pre_hook(hook, with_kwargs=True)
        real = qwen4_exp.qsa_indexer_kl_loss

        def record(inputs, query, key, **kwargs):
            loss = real(inputs, query, key, **kwargs)
            recorded.setdefault('calls', []).append((inputs, loss.detach()))
            return loss

        monkeypatch.setattr(qwen4_exp, 'qsa_indexer_kl_loss', record)
        _run(lm, _batch())

        calls = recorded['calls']
        assert len(calls) == len(_qsa_layers(lm)) == 2  # decoder QSA layer + MTP inner layer
        tracker = _tracker()
        for layer, (inputs, loss) in zip(_qsa_layers(lm), calls):
            q, k, mask = observed[id(layer)]  # [s, b, H, D]
            s, b = q.shape[:2]
            nb = s // R
            assert mask is not None and mask.dtype == torch.bool  # sbhd/CP1: TE mask (sparse) or causal (dense)
            allowed = ~mask[:, 0]
            if sparse:  # the bool selection is never the plain causal mask here
                assert not torch.equal(allowed, torch.ones(s, s, dtype=torch.bool, device='cuda').tril()[None])
            q_idx = inputs.q.detach().view(b, s, *inputs.q.shape[1:])
            keys = inputs.block_keys.detach().view(b, nb, inputs.block_keys.shape[-1])
            logits = _ref_block_logits(q_idx, keys)
            causal = _ref_causal(s, b, nb)
            support = _ref_topk_support(logits, causal, BUDGET // R) if sparse else causal
            hq, hkv = q.shape[2], k.shape[2]
            qh = q.permute(1, 2, 0, 3).float()
            kh = k.permute(1, 2, 0, 3).float().repeat_interleave(hq // hkv, dim=1)
            scores = (qh @ kh.transpose(-1, -2) * q.shape[-1]**-0.5).masked_fill(~allowed[:, None], float('-inf'))
            teacher = torch.softmax(scores, dim=-1).sum(dim=1)
            teacher = teacher / teacher.sum(-1, keepdim=True)
            ref = _ref_kl(logits, teacher, support, layer.self_attention.indexer.loss_coeff)
            torch.testing.assert_close(loss, ref, rtol=2e-5, atol=1e-7)
            assert loss.item() > 0
            torch.testing.assert_close(tracker[layer.layer_number - 1], loss)


def test_sparse_kl_trains_only_the_indexer():
    """Stage-2: the backbone sees the exact same forward and gradients as coeff=0; the indexer of
    every QSA layer (decoder + MTP) gets a gradient and a tracked loss."""
    batch = _batch()
    with _parallel_context():
        frozen, _ = _build(coeff=0.0)
        frozen_loss = _run(frozen, batch)
        frozen_backbone = _grads(frozen, indexer=False)
        assert _tracker() is None
        assert all(not p.requires_grad for n, p in frozen.named_parameters() if 'indexer' in n)

        trained, config = _build(coeff=0.5)
        assert config.experimental_attention_variant_loss_scale_func is DSAIndexerLossAutoScaler.set_loss_scale
        loss = _run(trained, batch)
        torch.testing.assert_close(loss, frozen_loss, rtol=0, atol=0)
        backbone = _grads(trained, indexer=False)
        assert backbone.keys() == frozen_backbone.keys()
        for name, grad in frozen_backbone.items():
            torch.testing.assert_close(backbone[name], grad, rtol=1e-5, atol=1e-6, msg=lambda m, n=name: f'{n}: {m}')
        indexer = _grads(trained, indexer=True)
        assert len(indexer) == 3 * len(_qsa_layers(trained))
        assert all(g.abs().sum() > 0 for g in indexer.values())
        tracker = _tracker()
        slots = {layer.layer_number - 1 for layer in _qsa_layers(trained)}
        assert all((tracker[i] > 0) == (i in slots) for i in range(tracker.numel()))


def test_dense_warmup_runs_full_attention():
    """Stage-1: the main attention is the full causal one the indexer is distilled from, i.e. the
    LM loss equals a model whose selection never engages (budget above the sequence)."""
    batch = _batch()
    with _parallel_context():
        full, _ = _build(coeff=0.0, budget=4096)
        full_loss = _run(full, batch)
        sparse, _ = _build(coeff=0.0)
        sparse_loss = _run(sparse, batch)
        assert not torch.equal(full_loss, sparse_loss)  # the selection does change the forward
        dense, _ = _build(coeff=0.5, sparse=False)
        torch.testing.assert_close(_run(dense, batch), full_loss, rtol=0, atol=0)
        assert all(g.abs().sum() > 0 for g in _grads(dense, indexer=True).values())


@pytest.mark.parametrize('sparse', [True, False])
@pytest.mark.parametrize('recompute', ['full', 'core_attn'])
def test_recompute_matches_and_logs_once(sparse, recompute):
    batch = _batch()
    with _parallel_context():
        plain, _ = _build(sparse=sparse)
        _run(plain, batch)
        ref_tracker, ref_grads = _tracker(), _grads(plain, indexer=True)
        DSAIndexerLossLoggingHelper.tracker = {}
        if recompute == 'full':
            overrides = dict(recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1)
        else:
            overrides = dict(recompute_granularity='selective', recompute_modules=['core_attn'])
        rerun, _ = _build(sparse=sparse, **overrides)
        _run(rerun, batch)
        torch.testing.assert_close(_tracker(), ref_tracker, rtol=1e-6, atol=1e-8)  # once, not twice
        grads = _grads(rerun, indexer=True)
        assert grads.keys() == ref_grads.keys()
        for name, grad in ref_grads.items():
            torch.testing.assert_close(grads[name], grad, rtol=1e-4, atol=1e-6, msg=lambda m, n=name: f'{n}: {m}')


def test_main_loss_scale_reaches_the_indexer_kl():
    """With the schedule's hook, two identical microbatches give the gradient of one; without it the
    indexer KL is not divided by num_microbatches."""
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    batch = _batch(b=1)

    def step(lm, num_microbatches):
        lm.zero_grad(set_to_none=True)

        def forward_step(data_iterator, model):
            ids, position_ids, mask, labels, loss_mask = next(data_iterator)
            out = model(ids, position_ids, mask, labels=labels, loss_mask=loss_mask)
            return out, lambda t: (t.float().mean(), {'loss': t.float().mean().detach()})

        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter([batch] * num_microbatches),
            model=[lm],
            num_microbatches=num_microbatches,
            seq_length=batch[0].shape[1],
            micro_batch_size=1,
            forward_only=False)
        return _grads(lm, indexer=True)

    with _parallel_context():
        lm, config = _build()
        one = step(lm, 1)
        two = step(lm, 2)
        for name, grad in one.items():
            torch.testing.assert_close(two[name], grad, rtol=1e-4, atol=1e-7, msg=lambda m, n=name: f'{n}: {m}')
        config.experimental_attention_variant_loss_scale_func = None  # negative control
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        unscaled = step(lm, 2)
        name = next(iter(one))
        torch.testing.assert_close(unscaled[name], 2 * one[name], rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize('sparse', [True, False])
def test_kl_decreases_when_training_the_indexer(sparse):
    batch = _batch()
    with _parallel_context():
        lm, _ = _build(coeff=1.0, sparse=sparse)
        params = [p for n, p in lm.named_parameters() if 'indexer' in n]
        for n, p in lm.named_parameters():
            if 'indexer' not in n:
                p.requires_grad_(False)
        opt = torch.optim.Adam(params, lr=3e-3)
        history = []
        for _ in range(30):
            DSAIndexerLossLoggingHelper.tracker = {}
            opt.zero_grad(set_to_none=True)
            _run(lm, batch)
            history.append(_tracker().sum().item())
            opt.step()
        assert history[-1] < 0.5 * history[0], history


def _packed_batch(lens, padded_lens=None, seed=29):
    g = torch.Generator(device='cuda').manual_seed(seed)
    padded_lens = padded_lens or lens
    T = sum(padded_lens)
    ids = torch.randint(1, 500, (1, T), device='cuda', generator=g)
    position_ids = torch.cat([torch.arange(n) for n in padded_lens]).cuda().unsqueeze(0)
    labels = torch.randint(1, 500, (1, T), device='cuda', generator=g)
    loss_mask = torch.ones(1, T, device='cuda', dtype=torch.bool)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device='cuda')
    cu_padded = torch.tensor([0, *torch.tensor(padded_lens).cumsum(0).tolist()], dtype=torch.int32, device='cuda')
    params = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_padded if padded_lens != lens else None,
        cu_seqlens_kv_padded=cu_padded if padded_lens != lens else None,
        max_seqlen_q=max(padded_lens),
        max_seqlen_kv=max(padded_lens))
    return (ids, position_ids, None, labels, loss_mask), params


@pytest.mark.parametrize('sparse', [True, False])
def test_packed_thd_documents_match_per_document_runs(sparse):
    """thd through the sparse kernel: the KL of a pack equals the row-weighted KLs of each document
    run alone (no cross-document teacher or support), and every QSA indexer trains."""
    lens = [96, 40]  # one sparse document, one within the budget
    with _parallel_context():
        lm, _ = _build(sparse=sparse, mtp=0, calculate_per_token_loss=True)
        batch, params = _packed_batch(lens)
        _run(lm, batch, params)
        packed = _tracker()
        assert packed.sum() > 0
        assert all(g.abs().sum() > 0 for g in _grads(lm, indexer=True).values())

        per_doc = torch.zeros_like(packed)
        offset = 0
        ids, position_ids, _, labels, loss_mask = batch
        for n in lens:
            DSAIndexerLossLoggingHelper.tracker = {}
            doc = slice(offset, offset + n)
            doc_params = PackedSeqParams(
                qkv_format='thd',
                cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32, device='cuda'),
                cu_seqlens_kv=torch.tensor([0, n], dtype=torch.int32, device='cuda'),
                max_seqlen_q=n,
                max_seqlen_kv=n)
            _run(lm, (ids[:, doc], position_ids[:, doc], None, labels[:, doc], loss_mask[:, doc]), doc_params)
            per_doc += _tracker()
            offset += n
        torch.testing.assert_close(packed, per_doc, rtol=2e-3, atol=1e-6)


def test_packed_padding_rows_are_excluded(monkeypatch):
    """A padded pack (cu_seqlens_q_padded) hands the KL exactly the real rows; the padding rows'
    exclusion itself is pinned numerically in test_qsa_indexer_loss.py."""
    seen = []
    real = qwen4_exp.qsa_indexer_kl_loss

    def record(inputs, query, key, **kwargs):
        seen.append((inputs.cu_seqlens.clone(), kwargs['valid_rows']))
        return real(inputs, query, key, **kwargs)

    monkeypatch.setattr(qwen4_exp, 'qsa_indexer_kl_loss', record)
    with _parallel_context():
        lm, _ = _build(mtp=0)
        batch, params = _packed_batch([90, 37], padded_lens=[96, 40])
        _run(lm, batch, params)
    assert len(seen) == 1
    cu, valid = seen[0]
    assert cu.tolist() == [0, 96, 136]  # documents on the padded grid the kernel attends
    expected = torch.ones(136, dtype=torch.bool, device='cuda')
    expected[90:96] = False
    expected[133:] = False
    assert torch.equal(valid, expected)
    assert torch.isfinite(_tracker()).all() and _tracker().sum() > 0


def test_frozen_indexer_skips_the_kl():
    """coeff > 0 with the indexer frozen afterwards (e.g. by a tuner): no KL, no tracker, same forward."""
    batch = _batch()
    with _parallel_context():
        reference, _ = _build(coeff=0.0)
        ref_loss = _run(reference, batch)
        lm, _ = _build(coeff=0.5)
        for name, p in lm.named_parameters():
            if 'indexer' in name:
                p.requires_grad_(False)
        torch.testing.assert_close(_run(lm, batch), ref_loss, rtol=0, atol=0)
        assert _tracker() is None
