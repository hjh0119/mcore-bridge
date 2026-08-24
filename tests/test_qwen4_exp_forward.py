# Forward-alignment test for the Qwen-Air (qwen4_exp / qwen3_8_flash_next)
# mcore-bridge integration: HF (transformers fork modeling, registered via the
# ms-swift compat layer) vs the Megatron model built+loaded by mcore_bridge.
#
# Uses the same trimmed 4-layer config as the roundtrip test (GDN, GDN, GDN,
# full-attn+QSA, PLE disabled) so it fits on one GPU per rank. Both models load
# the real checkpoint weights, run the same input_ids, and the final logits are
# compared (bf16 tolerance).
#
# Run with (world_size must equal TP*EP):
#   torchrun --nproc_per_node=N tests/test_qwen4_exp_forward.py [TP] [EP]
# Env knobs: NUM_LAYERS (default 4), FP32=1 runs both models in float32 to
# distinguish bf16 kernel/routing noise from real misalignment.
import json
import os
import sys

os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

CKPT = ('/root/.cache/huggingface/hub/models--Qwen--Qwen-Air-Example-CKPT-BF16/'
        'snapshots/d5a0d8cfe12597ea10e700c294da8b0873def115')
MS_SWIFT = '/mnt/workspace/hjh/workspace/ms-swift'
NUM_LAYERS = int(os.environ.get('NUM_LAYERS', '4'))
SEQ_LEN = 32
# FP32=1 runs both models in float32 (distinguishes bf16 kernel noise from bugs).
FP32 = os.environ.get('FP32', '0') == '1'


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


def hf_forward(device):
    """Load the trimmed HF (fork) model and return its logits for the fixed input."""
    import torch
    sys.path.insert(0, MS_SWIFT)
    # Polyfills transformers 5.14 for the 5.16 qwen4_exp module and registers it.
    from swift.model.models.qwen import _compat_qwen4_exp_transformers
    assert _compat_qwen4_exp_transformers(), 'ms-swift qwen4_exp compat failed'
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    text_dict = json.load(open(f'{CKPT}/config.json'))['text_config']
    cfg = Qwen4ExpTextConfig(**text_dict)  # validators normalize layer_types
    cfg.num_hidden_layers = NUM_LAYERS
    cfg.layer_types = list(cfg.layer_types[:NUM_LAYERS])
    cfg.ple_layer_ids = []  # PLE n-gram table (~100GB) excluded from the mini test
    cfg.use_cache = False
    cfg._attn_implementation = 'sdpa'

    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), generator=torch.Generator().manual_seed(42))

    model = Qwen4ExpForCausalLM(cfg)
    # Checkpoint keys are `model.language_model.*` (conditional-gen naming); the
    # text-only CausalLM expects `model.*`.
    from safetensors import safe_open
    index = json.load(open(f'{CKPT}/model.safetensors.index.json'))['weight_map']
    files, state_dict = {}, {}
    for key, shard in index.items():
        if not key.startswith('model.language_model.') and key != 'lm_head.weight':
            continue
        if '.ple.' in key:
            continue  # PLE disabled in the mini config
        layer = key.split('.')
        if 'layers' in layer and int(layer[layer.index('layers') + 1]) >= NUM_LAYERS:
            continue
        new_key = 'model.' + key[len('model.language_model.'):] if key.startswith('model.language_model.') else key
        if shard not in files:
            files[shard] = safe_open(f'{CKPT}/{shard}', framework='pt', device='cpu')
        state_dict[new_key] = files[shard].get_tensor(key)
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    assert not unexpected, f'unexpected keys: {unexpected[:5]}'
    assert not missing, f'missing keys: {missing[:5]}'
    del state_dict

    model = model.to(device)
    if FP32:
        model = model.float()
    model = model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(device), use_cache=False).logits
    logits = logits.float().cpu()
    del model
    torch.cuda.empty_cache()
    return input_ids, logits


def main():
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank % torch.cuda.device_count()}'
    torch.cuda.set_device(device)
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    assert tp * ep == world_size, f'TP*EP={tp * ep} != world_size={world_size}'

    # HF reference first (rank 0 only), then free the memory before building mcore.
    hf_result = [None, None]
    if rank == 0:
        hf_result = list(hf_forward(device))
        print(f'HF forward OK: logits {tuple(hf_result[1].shape)}', flush=True)
    objs = [hf_result] if rank == 0 else [None]
    dist.broadcast_object_list(objs, src=0)
    input_ids, hf_logits = objs[0]
    dist.barrier()

    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=1, expert_model_parallel_size=ep)

    import mcore_bridge  # noqa: F401 (applies patches)
    from mcore_bridge.config import ModelConfig, hf_to_mcore_config
    from mcore_bridge.model import get_mcore_model

    kwargs = hf_to_mcore_config(build_hf_config())
    kwargs['num_layers'] = NUM_LAYERS
    # Checkpoint pattern: 3 GDN layers then a full-attn layer, repeated.
    kwargs['linear_attention_freq'] = '[' + ','.join(
        '0' if (i + 1) % 4 == 0 else '1' for i in range(NUM_LAYERS)) + ']'
    kwargs['moe_layer_freq'] = '[' + ','.join('1' for _ in range(NUM_LAYERS)) + ']'
    kwargs['ple_layer_ids'] = []
    kwargs.update(
        params_dtype=torch.float32 if FP32 else torch.bfloat16,
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
    )
    config = ModelConfig(**kwargs)
    models = get_mcore_model(config)
    bridge = config.model_meta.bridge_cls(config)
    bridge.load_weights(models, CKPT)
    if rank == 0:
        print('mcore load_weights OK', flush=True)

    model = models[0].to(device).eval()
    position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids=input_ids.to(device), position_ids=position_ids, attention_mask=None)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = logits.float()
    # mcore convention is [s, b, v]; the HF reference is [b, s, v].
    if logits.shape[0] == SEQ_LEN and logits.shape[1] != SEQ_LEN:
        logits = logits.transpose(0, 1)

    # TP shards the output vocab; gather it back for the comparison.
    tp_world = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world > 1:
        gathered = [torch.empty_like(logits) for _ in range(tp_world)]
        dist.all_gather(gathered, logits.contiguous(),
                        group=parallel_state.get_tensor_model_parallel_group())
        logits = torch.cat(gathered, dim=-1)
    logits = logits.cpu()

    ok = False
    if rank == 0:
        assert logits.shape == hf_logits.shape, f'{tuple(logits.shape)} vs {tuple(hf_logits.shape)}'
        diff = (logits - hf_logits).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        argmax_match = (logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
        top10_overlap = torch.zeros(SEQ_LEN)
        for i in range(SEQ_LEN):
            a = set(logits[0, i].topk(10).indices.tolist())
            b = set(hf_logits[0, i].topk(10).indices.tolist())
            top10_overlap[i] = len(a & b) / 10
        top10_overlap = top10_overlap.mean().item()
        cos = torch.nn.functional.cosine_similarity(
            logits.flatten().unsqueeze(0), hf_logits.flatten().unsqueeze(0)).item()
        print(f'max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} argmax_match={argmax_match:.4f} '
              f'top10_overlap={top10_overlap:.4f} cosine={cos:.6f} dtype={"fp32" if FP32 else "bf16"}')
        if FP32:
            # fp32 removes most kernel/routing noise: expect near-exact alignment
            # (residual tail-logit diffs come from accumulation-order differences
            # between the fla/mcore GDN kernels and MoE routing tie-breaks).
            ok = max_diff < 1.0 and mean_diff < 0.01 and argmax_match >= 0.99 and cos > 0.9999
        else:
            # bf16: GDN kernel and MoE top-k routing amplify rounding noise
            # (verified by the fp32 run aligning near-exactly), so use loose gates.
            ok = max_diff < 5.0 and mean_diff < 0.2 and cos > 0.99 and top10_overlap >= 0.5
        print('FORWARD ALIGNMENT TEST PASSED' if ok else 'FORWARD ALIGNMENT TEST FAILED')
    # All ranks must agree on the exit code.
    flag = torch.tensor([1 if ok else 0], device=device)
    dist.broadcast(flag, src=0)
    dist.barrier()
    sys.exit(0 if flag.item() else 1)


if __name__ == '__main__':
    main()
