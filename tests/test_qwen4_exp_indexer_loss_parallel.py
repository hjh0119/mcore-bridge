# Copyright (c) ModelScope Contributors. All rights reserved.
"""QSA indexer KL under real NCCL tensor / context parallelism.

* ``qsa_indexer_kl_loss`` itself: TP head-sharded teacher (summed over the TP group) and CP
  row-sharded queries reproduce the single-rank loss and gradients; negative controls drop the
  head all-reduce.
* The whole tiny Qwen3.8-Flash-Next model: TP=2 with SP and CP=2 (sbhd and thd)
  reproduce the single-rank indexer KL and indexer gradients from the same weights (exported
  from the single-rank model and loaded back through the bridge).
"""
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core import parallel_state

requires_two_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason='requires 2 CUDA devices and NCCL')

R, BUDGET, HQ, HKV, DM = 4, 32, 4, 2, 32


def _init(rank, world_size, init_file):
    # TE / cuBLAS default fp32 GEMMs to TF32; pin true fp32 so single- vs multi-rank compare tightly.
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method=f'file://{init_file}', rank=rank, world_size=world_size)


def _teardown():
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------- the loss function
def _loss_inputs(device):
    from mcore_bridge.model.modules import qsa_indexer as qi
    g = torch.Generator(device='cpu').manual_seed(3)
    lens = [96, 40, 8]  # every length is a multiple of 2 * cp for the zigzag split
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=device)
    T = sum(lens)
    nb = sum(n // R for n in lens)
    q_idx = torch.randn(T, 4, 16, generator=g).to(device)
    keys = torch.randn(nb, 16, generator=g).to(device)
    top, keep = qi._score_packed_blocks(q_idx, keys, cu, R, BUDGET // R)
    main_q = (torch.randn(T, HQ, DM, generator=g) * 2).to(device)
    main_k = (torch.randn(T, HKV, DM, generator=g) * 2).to(device)
    return cu, q_idx, keys, top, keep, main_q, main_k


def _kl(cu, q_idx, keys, top, keep, query, key, sparse, rows=None, tp_group=None, per_token=False):
    from mcore_bridge.model.modules.qsa_indexer import QSALossInputs
    from mcore_bridge.model.modules.qsa_indexer_loss import qsa_indexer_kl_loss
    q_leaf = q_idx.clone().requires_grad_(True)
    k_leaf = keys.clone().requires_grad_(True)
    inputs = QSALossInputs(q_leaf, k_leaf, cu, top if sparse else None, keep if sparse else None)
    loss = qsa_indexer_kl_loss(
        inputs,
        query,
        key,
        softmax_scale=DM**-0.5,
        compress_ratio=R,
        loss_coeff=0.5,
        query_rows=rows,
        tp_group=tp_group,
        calculate_per_token_loss=per_token)
    loss.backward()
    return loss.detach(), q_leaf.grad, k_leaf.grad


def _loss_worker(rank, world_size, init_file):
    from mcore_bridge.model.modules.kernels.qsa_kernels import _cp_query_global_positions_thd
    try:
        _init(rank, world_size, init_file)
        device = torch.device('cuda', rank)
        cu, q_idx, keys, top, keep, main_q, main_k = _loss_inputs(device)
        group = dist.group.WORLD
        for sparse in (True, False):
            ref = _kl(cu, q_idx, keys, top, keep, main_q, main_k, sparse)
            # TP: this rank owns HQ / world query heads and their kv heads
            hq, hkv = HQ // world_size, HKV // world_size
            local_q = main_q[:, rank * hq:(rank + 1) * hq]
            local_k = main_k[:, rank * hkv:(rank + 1) * hkv]
            tp = _kl(cu, q_idx, keys, top, keep, local_q, local_k, sparse, tp_group=group)
            for got, want in zip(tp, ref):
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)
            # negative control: without the head all-reduce the teacher is a partial head sum
            partial = _kl(cu, q_idx, keys, top, keep, local_q, local_k, sparse)
            assert not torch.allclose(partial[0], ref[0], rtol=1e-3)
            # CP: zigzag rows; mean mode averages, per-token mode sums
            rows = _cp_query_global_positions_thd(cu, world_size, rank, device)
            for per_token, op in ((False, dist.ReduceOp.AVG), (True, dist.ReduceOp.SUM)):
                full = _kl(cu, q_idx, keys, top, keep, main_q, main_k, sparse, per_token=per_token)
                cp = list(_kl(cu, q_idx, keys, top, keep, main_q[rows], main_k, sparse, rows=rows, per_token=per_token))
                for t in cp:
                    dist.all_reduce(t, op=op)
                for got, want in zip(cp, full):
                    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)
    finally:
        _teardown()


@requires_two_gpus
def test_kl_tp_heads_and_cp_rows_match_single_rank(tmp_path):
    mp.spawn(_loss_worker, args=(2, str(tmp_path / 'kl-init')), nprocs=2, join=True)


# ---------------------------------------------------------------- the whole model
def _model_config(tp=1, cp=1, sp=False, sparse=True, dtype=torch.float32):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_qwen4_exp_mtp import _tiny_qwen4exp_config

    from mcore_bridge.config import ModelConfig
    from mcore_bridge.config.parser import hf_to_mcore_config
    hf_config = _tiny_qwen4exp_config()
    hf_config.text_config.indexer_budget = BUDGET
    values = hf_to_mcore_config(hf_config)
    values.update(
        mcore_model_type='qwen4_exp',
        hf_config=hf_config,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        perform_initialization=True,
        use_cpu_initialization=False,
        language_model_only=True,
        moe_grouped_gemm=True,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        cp_comm_type='all_gather' if cp > 1 else None,
        expert_tensor_parallel_size=tp,
        sequence_parallel=sp,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        dsa_indexer_loss_coeff=0.5,
        dsa_indexer_use_sparse_loss=sparse,
    )
    return ModelConfig(**values)


def _build(config, seed=5):
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from mcore_bridge.model.register import get_mcore_model
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    model = get_mcore_model(config)[0].cuda()
    return model, model.language_model if hasattr(model, 'language_model') else model


def _batch(layout, device):
    from megatron.core.packed_seq_params import PackedSeqParams
    g = torch.Generator(device='cpu').manual_seed(23)
    if layout == 'thd':
        lens = [96, 40]
        T = sum(lens)
        cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
        packed = PackedSeqParams(
            qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
        ids = torch.randint(1, 500, (1, T), generator=g).to(device)
        position_ids = torch.cat([torch.arange(n) for n in lens]).to(device).unsqueeze(0)
    else:
        packed = None
        ids = torch.randint(1, 500, (2, 96), generator=g).to(device)
        position_ids = torch.arange(96, device=device).unsqueeze(0).expand(2, -1)
    labels = torch.randint(1, 500, ids.shape, generator=g).to(device)
    return ids, position_ids, labels, packed


def _forward_backward(lm, ids, position_ids, labels, packed, mask=None):
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
    DSAIndexerLossLoggingHelper.tracker = {}
    lm.train()
    lm.zero_grad(set_to_none=True)
    out = lm(
        ids,
        position_ids,
        mask,
        labels=labels,
        loss_mask=torch.ones_like(labels, dtype=torch.bool),
        packed_seq_params=packed)
    out.float().mean().backward()
    grads = {n: p.grad.detach().float().clone() for n, p in lm.named_parameters() if 'indexer' in n}
    return DSAIndexerLossLoggingHelper.tracker['values'].clone(), grads


def _export(config, model):
    out = {}
    for k, v in config.bridge._convert([model], {}, '', False, 'Exporting test: '):
        out[k] = v.load() if hasattr(v, 'load') else v
    return out


class _Lazy:

    def __init__(self, t):
        self.t = t

    def load(self):
        return self.t


def _model_worker(rank, world_size, init_file, mode, sparse):
    from mcore_bridge.utils.megatron_utils import split_cp_inputs
    try:
        _init(rank, world_size, init_file)
        device = torch.device('cuda', rank)
        layout = 'thd' if mode == 'cp_thd' else 'sbhd'
        tp = world_size if mode.startswith('tp') else 1
        cp = world_size if mode.startswith('cp') else 1
        # fp32 keeps the comparison tight; TE has no fp32 backend for dense attention under CP.
        dtype = torch.bfloat16 if cp > 1 and not sparse else torch.float32
        ids, position_ids, labels, packed = _batch(layout, device)
        mask = None
        if layout == 'sbhd':
            s = ids.shape[1]
            mask = torch.triu(torch.ones(ids.shape[0], 1, s, s, dtype=torch.bool, device=device), diagonal=1)

        # single-rank reference (every rank runs the same thing, dp=2)
        parallel_state.initialize_model_parallel()
        ref_config = _model_config(sparse=sparse, dtype=dtype)
        ref_model, ref_lm = _build(ref_config)
        ref_tracker, ref_grads = _forward_backward(ref_lm, ids, position_ids, labels, packed, mask)
        weights = _export(ref_config, ref_model)
        decoder_slot = next(layer.layer_number - 1 for layer in ref_lm.decoder.layers
                            if getattr(layer.self_attention, 'indexer', None) is not None)
        del ref_model, ref_lm
        parallel_state.destroy_model_parallel()
        from mcore_bridge.bridge import gpt_bridge
        gpt_bridge.EP_PP_GROUP = gpt_bridge.EP_PP_RANK = gpt_bridge.EP_PP_SIZE = None

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp, context_parallel_size=cp, expert_tensor_parallel_size=tp)
        config = _model_config(tp=tp, cp=cp, sp=mode == 'tp_sp', sparse=sparse, dtype=dtype)
        model, lm = _build(config, seed=999)
        list(config.bridge._convert([model], {k: _Lazy(v) for k, v in weights.items()}, '', True, 'Loading test: '))
        if cp > 1:
            cu = None if packed is None else packed.cu_seqlens_q
            ids, position_ids, labels = (split_cp_inputs(t, cu, dim=1) for t in (ids, position_ids, labels))
            mask = None  # causal under CP
        tracker, grads = _forward_backward(lm, ids, position_ids, labels, packed, mask)
        if cp > 1:  # each CP rank owns its rows: the row mean averages over the CP group
            group = parallel_state.get_context_parallel_group()
            dist.all_reduce(tracker, op=dist.ReduceOp.AVG, group=group)
            for g in grads.values():
                dist.all_reduce(g, op=dist.ReduceOp.AVG, group=group)
        case = f'{mode} sparse={sparse} {dtype} rank={rank}'
        slots = ref_tracker.nonzero().flatten().tolist()
        assert len(slots) == 2, f'{case}: expected the decoder and MTP QSA layers, got {slots}'
        if cp > 1:
            # Under CP the MTP inner layer already receives different rotary angles than at cp=1
            # (its main-attention keys differ before any KL code runs), so only the decoder QSA
            # layer is a valid CP oracle.
            slots = [decoder_slot]
            grads = {n: g for n, g in grads.items() if n.startswith('decoder.')}
            ref_grads = {n: g for n, g in ref_grads.items() if n.startswith('decoder.')}
        kl_rtol, grad_rel = (1e-4, 1e-3) if dtype == torch.float32 else (1e-2, 5e-2)
        torch.testing.assert_close(
            tracker[slots], ref_tracker[slots], rtol=kl_rtol, atol=1e-6, msg=lambda m: f'{case} KL: {m}')
        assert grads.keys() == ref_grads.keys() and grads
        for name, ref in ref_grads.items():
            err = ((grads[name] - ref).norm() / ref.norm()).item()
            assert err < grad_rel, f'{case} {name}: relative L2 error {err:.3e}'
        if rank == 0:
            print(f'{case}: KL {tracker[slots].tolist()} vs {ref_tracker[slots].tolist()}')
    finally:
        _teardown()


@requires_two_gpus
# MoE refuses TP without SP in training, so TP is covered as TP+SP
@pytest.mark.parametrize('mode', ['tp_sp', 'cp_sbhd', 'cp_thd'])
@pytest.mark.parametrize('sparse', [True, False])
def test_model_parallel_kl_matches_single_rank(tmp_path, mode, sparse):
    mp.spawn(_model_worker, args=(2, str(tmp_path / 'model-init'), mode, sparse), nprocs=2, join=True)
