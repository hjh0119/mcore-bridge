# Roundtrip conversion test for the Qwen-Air (qwen4_exp / qwen3_8_flash_next)
# mcore-bridge integration: HF -> mcore -> HF weight equivalence.
#
# The full checkpoint is ~125B params (512 experts x 48 layers) and the PLE
# n-gram table alone is ~100GB (TP-sharded in production), so this test builds
# a trimmed 4-layer model (layers 0-3: GDN, GDN, GDN+PLE, full-attn+QSA) and
# checks that every mapped HF key roundtrips bitwise. MTP keys are out of
# scope and excluded. Modes:
#   nople (default): full-size experts, PLE disabled -> exact comparison.
#   ple: shrunk n-gram table so it fits on one GPU; the shard rows loaded are
#        a prefix of the checkpoint shards and are compared as such.
# Run with (world_size must equal TP*EP*PP):
#   torchrun --nproc_per_node=N tests/test_qwen4_exp_roundtrip.py TP EP MODE [SP] [PP] [SAVE_DIR]
# SP=1 enables sequence_parallel (needs TP>1); SAVE_DIR additionally runs the
# production save_weights() path and verifies the files written to disk.
import json
import os
import sys

os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

CKPT = ('/root/.cache/huggingface/hub/models--Qwen--Qwen-Air-Example-CKPT-BF16/'
        'snapshots/d5a0d8cfe12597ea10e700c294da8b0873def115')
NUM_LAYERS = 4


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


def _layer_number(key: str):
    import re
    m = re.match(r'(?:model\.language_model\.)?layers\.(\d+)\.', key)
    return int(m.group(1)) if m else None


def _skip_key(key: str, mode: str):
    """Reasons to skip a checkpoint key in the comparison, or None."""
    if key.startswith('mtp.'):
        return 'mtp'  # MTP out of scope
    if mode == 'nople' and '.ple.' in key:
        return 'ple'  # PLE disabled in this mode
    layer_no = _layer_number(key)
    if layer_no is not None and layer_no >= NUM_LAYERS:
        return 'trim'  # trimmed model only builds the first NUM_LAYERS layers
    return None


def main():
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    mode = sys.argv[3] if len(sys.argv) > 3 else 'nople'
    sp = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    pp = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    save_dir = sys.argv[6] if len(sys.argv) > 6 else None
    assert mode in {'nople', 'ple'}
    assert tp * ep * pp == world_size, f'TP*EP*PP={tp * ep * pp} != world_size={world_size}'

    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep)

    import mcore_bridge  # noqa: F401 (applies patches)
    from mcore_bridge.config import ModelConfig, hf_to_mcore_config
    from mcore_bridge.model import get_mcore_model

    kwargs = hf_to_mcore_config(build_hf_config())
    # Trim to the first NUM_LAYERS layers to fit in memory: keeps one of each
    # layer kind (GDN, GDN+PLE, full-attn) and all global modules.
    kwargs['num_layers'] = NUM_LAYERS
    kwargs['linear_attention_freq'] = '[1,1,1,0]'
    kwargs['moe_layer_freq'] = '[1,1,1,1]'
    if mode == 'nople':
        kwargs['ple_layer_ids'] = []
    else:
        kwargs['ple_layer_ids'] = [i for i in kwargs.get('ple_layer_ids', []) if i <= NUM_LAYERS]
        # Shrink the n-gram table (~100GB at checkpoint size); the conversion
        # mapping is size-agnostic and shard rows stay a prefix of the ckpt's.
        kwargs['ngram_vocab_size_base'] = 65536
    kwargs.update(
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        pipeline_model_parallel_size=pp,
        sequence_parallel=bool(sp),
        context_parallel_size=1,
    )
    config = ModelConfig(**kwargs)
    if rank == 0:
        print(f'mode={mode}, mcore_model_type: {config.mcore_model_type}, '
              f'TP={tp}, EP={ep}, PP={pp}, SP={sp}', flush=True)

    models = get_mcore_model(config)
    bridge = config.model_meta.bridge_cls(config)
    bridge.load_weights(models, CKPT)
    if rank == 0:
        print('load_weights OK', flush=True)

    # Production save path (only_master_rank + streaming safetensor saver).
    if save_dir:
        bridge.save_weights(models, save_dir)
        if rank == 0:
            print(f'save_weights OK -> {save_dir}', flush=True)

    exported = {}
    for k, v in bridge.export_weights(models, target_device='cpu'):
        if k in exported:
            raise RuntimeError(f'duplicate exported key: {k}')
        exported[k] = v
    if rank == 0:
        print(f'export_weights OK: {len(exported)} keys', flush=True)

    # Bitwise roundtrip comparison against the HF checkpoint (rank 0 only).
    ok = True
    if rank == 0:
        from safetensors import safe_open
        index = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
        files = {}
        n_checked = n_skipped = 0
        failures = []
        for key, shard in index.items():
            if _skip_key(key, mode):
                n_skipped += 1
                continue
            if key not in exported:
                failures.append(f'MISSING in export: {key}')
                continue
            if shard not in files:
                files[shard] = safe_open(f'{CKPT}/{shard}', framework='pt', device='cpu')
            ref = files[shard].get_tensor(key)
            got = exported[key]
            if mode == 'ple' and 'ngram_embedding.shard_' in key:
                # Shrunk table: only the first rows were loaded/exported.
                ref = ref[:got.shape[0]]
            if ref.shape != got.shape:
                failures.append(f'SHAPE {key}: hf {tuple(ref.shape)} vs exported {tuple(got.shape)}')
                continue
            if ref.dtype != got.dtype:
                failures.append(f'DTYPE {key}: hf {ref.dtype} vs exported {got.dtype}')
                continue
            if not torch.equal(ref, got):
                diff = (ref.float() - got.float()).abs().max().item()
                failures.append(f'VALUE {key}: max_diff {diff}')
            n_checked += 1
        unexpected = [k for k in exported if k not in index]
        print(f'checked={n_checked} skipped={n_skipped} unexpected={len(unexpected)} '
              f'missing_or_bad={len(failures)}')
        for f in failures[:20]:
            print(' ', f)
        for k in unexpected[:20]:
            print('  UNEXPECTED:', k)
        ok = not failures and not unexpected
        print('ROUNDTRIP TEST PASSED' if ok else 'ROUNDTRIP TEST FAILED')

        if save_dir:
            # Verify the files written to disk by save_weights().
            if os.path.exists(f'{save_dir}/model.safetensors.index.json'):
                saved_map = json.load(open(f'{save_dir}/model.safetensors.index.json'))['weight_map']
            else:
                single = f'{save_dir}/model.safetensors'
                saved_map = {k: 'model.safetensors' for k in safe_open(single, framework='pt').keys()}
            sfiles = {}
            n_checked_s = n_skipped_s = 0
            fails = []
            for key, shard in index.items():
                if _skip_key(key, mode):
                    n_skipped_s += 1
                    continue
                if key not in saved_map:
                    fails.append(f'MISSING on disk: {key}')
                    continue
                if shard not in files:
                    files[shard] = safe_open(f'{CKPT}/{shard}', framework='pt', device='cpu')
                if saved_map[key] not in sfiles:
                    sfiles[saved_map[key]] = safe_open(f'{save_dir}/{saved_map[key]}',
                                                       framework='pt', device='cpu')
                ref = files[shard].get_tensor(key)
                got = sfiles[saved_map[key]].get_tensor(key)
                if mode == 'ple' and 'ngram_embedding.shard_' in key:
                    ref = ref[:got.shape[0]]
                if ref.shape != got.shape:
                    fails.append(f'SHAPE {key}: hf {tuple(ref.shape)} vs saved {tuple(got.shape)}')
                    continue
                if ref.dtype != got.dtype:
                    fails.append(f'DTYPE {key}: hf {ref.dtype} vs saved {got.dtype}')
                    continue
                if not torch.equal(ref, got):
                    diff = (ref.float() - got.float()).abs().max().item()
                    fails.append(f'VALUE {key}: max_diff {diff}')
                n_checked_s += 1
            unexpected_s = [k for k in saved_map if k not in index]
            print(f'[saved] checked={n_checked_s} skipped={n_skipped_s} unexpected={len(unexpected_s)} '
                  f'missing_or_bad={len(fails)}')
            for f in fails[:20]:
                print(' ', f)
            for k in unexpected_s[:20]:
                print('  UNEXPECTED:', k)
            ok &= not fails and not unexpected_s
            print('SAVE TEST PASSED' if not fails and not unexpected_s else 'SAVE TEST FAILED')
    dist.barrier()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
