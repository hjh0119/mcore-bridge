# Parallel-combination forward test for Qwen-Air (qwen4_exp / qwen3_8_flash_next).
#
# The single-GPU alignment against the HF fork is covered by
# tests/test_qwen4_exp_forward.py. This test reuses that comparison and asserts
# that every TP/EP/PP/SP combination reproduces the same HF reference logits,
# i.e. that the parallel plumbing (HC replicated weights + TP grad-reduce,
# GDN/QSA sharding, MoE expert parallel, PLE replication, HC-mixer on the last
# PP stage) does not change the model output.
#
# Run one combination:
#   torchrun --nproc_per_node=4 tests/test_qwen4_exp_parallel_forward.py --tp 2 --ep 2
# Run the whole matrix (spawns one torchrun per combination):
#   python tests/test_qwen4_exp_parallel_forward.py --matrix
#
# Coverage notes:
#   - PLE is OFF by default. Its n-gram table is ~320M rows (~205GB here) so it
#     cannot be materialized, and shrinking `ngram_vocab_size_base` re-derives
#     different per-head primes, which makes the checkpoint's hash buffers and
#     rows inconsistent with the shrunk table (out-of-range gather indices). PLE
#     under parallelism is instead covered by tests/test_qwen4_exp_roundtrip.py,
#     which takes TP/EP/PP/SP plus a `ple` mode and checks PLE weight conversion
#     bitwise. Pass --ple here only with a config whose table actually fits.
#   - With PP>1 only the last stage holds the LM head, so that stage compares.
import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

CKPT = ('/root/.cache/huggingface/hub/models--Qwen--Qwen-Air-Example-CKPT-BF16/'
        'snapshots/d5a0d8cfe12597ea10e700c294da8b0873def115')
MS_SWIFT = '/mnt/workspace/hjh/workspace/ms-swift'
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', '4'))
SEQ_LEN = 32

# (tp, ep, pp, sp); world_size = tp*ep*pp
MATRIX = [
    (1, 1, 1, 0),
    (2, 1, 1, 0),
    (1, 2, 1, 0),
    (2, 2, 1, 0),
    (1, 1, 2, 0),
    (2, 1, 1, 1),  # SP (PLE is asserted incompatible with SP)
]


def build_hf_config():
    import torch
    from transformers import PretrainedConfig
    config_dict = json.load(open(f'{CKPT}/config.json'))
    cfg = PretrainedConfig.from_dict(config_dict)
    cfg.text_config = PretrainedConfig.from_dict(config_dict['text_config'])
    if 'vision_config' in config_dict:
        cfg.vision_config = PretrainedConfig.from_dict(config_dict['vision_config'])
        cfg.vision_config.dtype = torch.bfloat16
    cfg.name_or_path = CKPT
    return cfg


def hf_forward(device, use_ple, fp32):
    """HF (fork) reference logits for the fixed input, trimmed like the mcore model."""
    import torch
    sys.path.insert(0, MS_SWIFT)
    from swift.model.models.qwen import _compat_qwen4_exp_transformers
    assert _compat_qwen4_exp_transformers(), 'ms-swift qwen4_exp compat failed'
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    text_dict = json.load(open(f'{CKPT}/config.json'))['text_config']
    cfg = Qwen4ExpTextConfig(**text_dict)  # validators normalize layer_types
    cfg.num_hidden_layers = NUM_LAYERS
    cfg.layer_types = list(cfg.layer_types[:NUM_LAYERS])
    cfg.ple_layer_ids = [i for i in (cfg.ple_layer_ids or []) if i <= NUM_LAYERS] if use_ple else []
    cfg.use_cache = False
    cfg._attn_implementation = 'sdpa'

    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), generator=torch.Generator().manual_seed(42))

    model = Qwen4ExpForCausalLM(cfg)
    from safetensors import safe_open
    index = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
    files, state_dict = {}, {}
    for key, shard in index.items():
        if not key.startswith('model.language_model.') and key != 'lm_head.weight':
            continue
        parts = key.split('.')
        if 'layers' in parts and int(parts[parts.index('layers') + 1]) >= NUM_LAYERS:
            continue
        if '.ple.' in key and not use_ple:
            continue
        new_key = 'model.' + key[len('model.language_model.'):] if key.startswith('model.language_model.') else key
        if shard not in files:
            files[shard] = safe_open(f'{CKPT}/{shard}', framework='pt', device='cpu')
        state_dict[new_key] = files[shard].get_tensor(key)

    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    assert not unexpected, f'unexpected keys: {unexpected[:5]}'
    # The PLE n-gram hash buffers (multipliers / per-head prime sizes+offsets) are
    # derived from the config by the module itself, and both models derive them
    # from the same config, so they are expected to be absent from state_dict.
    ple_buffer_names = ('layer_multipliers', 'ngram_heads_vocab_sizes', 'ngram_heads_offsets')
    missing = [k for k in missing if not k.endswith(ple_buffer_names)]
    assert not missing, f'missing keys: {missing[:5]}'
    del state_dict

    model = model.to(device)
    if fp32:
        model = model.float()
    model = model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(device), use_cache=False).logits
    logits = logits.float().cpu()
    del model
    torch.cuda.empty_cache()
    return input_ids, logits


def run_one(tp, ep, pp, sp, use_ple, fp32):
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank % torch.cuda.device_count()}'
    torch.cuda.set_device(device)
    assert tp * ep * pp == world_size, f'TP*EP*PP={tp * ep * pp} != world_size={world_size}'
    if sp:
        assert tp > 1, 'sequence_parallel requires TP>1'
        # mcore-bridge asserts PLE is incompatible with SP.
        use_ple = False

    tag = f'TP{tp}/EP{ep}/PP{pp}/SP{sp}/PLE{int(use_ple)}'

    payload = [None]
    if rank == 0:
        payload = [list(hf_forward(device, use_ple, fp32))]
        print(f'[{tag}] HF reference OK: logits {tuple(payload[0][1].shape)}', flush=True)
    dist.broadcast_object_list(payload, src=0)
    input_ids, hf_logits = payload[0]
    dist.barrier()

    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, expert_model_parallel_size=ep)
    # sequence_parallel makes TransformerBlock fork the CUDA RNG tracker
    # (transformer_block.py: `get_cuda_rng_tracker().fork()`), which requires the
    # model-parallel rng state to exist. Training entrypoints do this for us.
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)

    import mcore_bridge  # noqa: F401 (applies patches)
    from mcore_bridge.config import ModelConfig, hf_to_mcore_config
    from mcore_bridge.model import get_mcore_model

    kwargs = hf_to_mcore_config(build_hf_config())
    kwargs['num_layers'] = NUM_LAYERS
    kwargs['linear_attention_freq'] = '[' + ','.join(
        '0' if (i + 1) % 4 == 0 else '1' for i in range(NUM_LAYERS)) + ']'
    kwargs['moe_layer_freq'] = '[' + ','.join('1' for _ in range(NUM_LAYERS)) + ']'
    kwargs['ple_layer_ids'] = ([i for i in (kwargs.get('ple_layer_ids') or []) if i <= NUM_LAYERS]
                               if use_ple else [])
    kwargs.update(
        params_dtype=torch.float32 if fp32 else torch.bfloat16,
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        pipeline_model_parallel_size=pp,
        sequence_parallel=bool(sp),
        context_parallel_size=1,
    )
    config = ModelConfig(**kwargs)
    models = get_mcore_model(config)
    bridge = config.model_meta.bridge_cls(config)
    bridge.load_weights(models, CKPT)
    if rank == 0:
        print(f'[{tag}] mcore load_weights OK', flush=True)

    assert len(models) == 1, f'unexpected virtual-pipeline split: {len(models)} chunks'
    model = models[0].to(device).eval()
    position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    ids = input_ids.to(device)
    dtype = torch.float32 if fp32 else torch.bfloat16

    if pp > 1:
        # Drive the pipeline manually: each stage consumes the previous stage's
        # hidden states, so only the last stage produces logits.
        ps = parallel_state
        if not ps.is_pipeline_first_stage():
            recv = torch.empty((SEQ_LEN, 1, config.hidden_size * config.hc_count),
                               dtype=dtype, device=device)
            dist.recv(recv, src=ps.get_pipeline_model_parallel_prev_rank())
            model.set_input_tensor(recv)
        with torch.no_grad():
            out = model(input_ids=ids, position_ids=position_ids, attention_mask=None)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not ps.is_pipeline_last_stage():
            dist.send(out.contiguous(), dst=ps.get_pipeline_model_parallel_next_rank())
            logits = None
        else:
            logits = out
    else:
        with torch.no_grad():
            logits = model(input_ids=ids, position_ids=position_ids, attention_mask=None)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

    ok_local = False
    if logits is not None:
        logits = logits.float()
        tp_world = parallel_state.get_tensor_model_parallel_world_size()
        if sp and tp_world > 1 and logits.shape[0] == SEQ_LEN // tp_world:
            # sequence_parallel keeps only this rank's slice of the sequence.
            seq_parts = [torch.empty_like(logits) for _ in range(tp_world)]
            dist.all_gather(seq_parts, logits.contiguous(),
                            group=parallel_state.get_tensor_model_parallel_group())
            logits = torch.cat(seq_parts, dim=0)
        # mcore emits [s, b, v]; the HF reference is [b, s, v].
        if logits.shape[0] == SEQ_LEN and logits.shape[1] != SEQ_LEN:
            logits = logits.transpose(0, 1)
        if tp_world > 1:
            gathered = [torch.empty_like(logits) for _ in range(tp_world)]
            dist.all_gather(gathered, logits.contiguous(),
                            group=parallel_state.get_tensor_model_parallel_group())
            logits = torch.cat(gathered, dim=-1)
        logits = logits.cpu()
        if parallel_state.get_tensor_model_parallel_rank() == 0:
            assert logits.shape == hf_logits.shape, f'{tuple(logits.shape)} vs {tuple(hf_logits.shape)}'
            diff = (logits - hf_logits).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            argmax_match = (logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
            cos = torch.nn.functional.cosine_similarity(
                logits.flatten().unsqueeze(0), hf_logits.flatten().unsqueeze(0)).item()
            if fp32:
                ok_local = max_diff < 1.0 and mean_diff < 0.01 and argmax_match >= 0.99 and cos > 0.9999
            else:
                ok_local = max_diff < 5.0 and mean_diff < 0.2 and cos > 0.99 and argmax_match >= 0.9
            print(f'[{tag}] max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} '
                  f'argmax_match={argmax_match:.4f} cosine={cos:.6f} '
                  f'dtype={"fp32" if fp32 else "bf16"} -> {"PASS" if ok_local else "FAIL"}', flush=True)

    # Only the comparing rank knows the verdict; MAX-reduce it to every rank.
    verdict = torch.tensor([1 if ok_local else 0], device=device, dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MAX)
    ok = verdict.item() > 0
    if rank == 0:
        print(f'[{tag}] {"PARALLEL FORWARD PASSED" if ok else "PARALLEL FORWARD FAILED"}', flush=True)
    dist.barrier()
    return ok


def run_matrix(fp32, use_ple):
    """Spawn one torchrun per combination and summarize."""
    listing = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout
    n_gpu = sum(1 for line in listing.splitlines() if line.startswith('GPU '))
    results = []
    for tp, ep, pp, sp in MATRIX:
        world = tp * ep * pp
        if world > n_gpu:
            results.append((tp, ep, pp, sp, f'SKIP(need {world} gpu, have {n_gpu})'))
            continue
        cmd = [
            sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={world}',
            '--master_port', str(29500 + world * 7 + tp * 3 + pp), __file__,
            '--tp', str(tp), '--ep', str(ep), '--pp', str(pp), '--sp', str(sp),
        ]
        if fp32:
            cmd.append('--fp32')
        if use_ple:
            cmd.append('--ple')
        print(f'\n===== RUN TP{tp} EP{ep} PP{pp} SP{sp} (world={world}) =====', flush=True)
        proc = subprocess.run(cmd, env=os.environ.copy())
        results.append((tp, ep, pp, sp, 'PASS' if proc.returncode == 0 else 'FAIL'))
    print('\n================ PARALLEL MATRIX SUMMARY ================')
    for tp, ep, pp, sp, status in results:
        print(f'  TP={tp} EP={ep} PP={pp} SP={sp}: {status}')
    failed = [r for r in results if r[4] == 'FAIL']
    print('ALL PARALLEL COMBINATIONS PASSED' if not failed else f'{len(failed)} COMBINATION(S) FAILED')
    return not failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--ep', type=int, default=1)
    parser.add_argument('--pp', type=int, default=1)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--fp32', action='store_true', help='run both models in fp32 (tight tolerances)')
    parser.add_argument('--ple', action='store_true',
                        help='enable the PLE layer (needs a config whose n-gram table fits; see notes)')
    parser.add_argument('--matrix', action='store_true', help='spawn the full combination matrix')
    args = parser.parse_args()
    if args.matrix:
        sys.exit(0 if run_matrix(args.fp32, args.ple) else 1)
    ok = run_one(args.tp, args.ep, args.pp, args.sp, args.ple, args.fp32)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
