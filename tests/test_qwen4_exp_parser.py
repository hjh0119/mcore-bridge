import json
import sys
import types

# Import mcore_bridge from src without running the package __init__ (which
# applies TE patches unavailable in this sandbox).
pkg = types.ModuleType('mcore_bridge')
pkg.__path__ = ['/mnt/workspace/hjh/mcore-bridge/src/mcore_bridge']
sys.modules['mcore_bridge'] = pkg

from transformers import PretrainedConfig

from mcore_bridge.config.parser import hf_to_mcore_config

CKPT = ('/root/.cache/huggingface/hub/models--Qwen--Qwen-Air-Example-CKPT-BF16/'
        'snapshots/d5a0d8cfe12597ea10e700c294da8b0873def115')

config_dict = json.load(open(f'{CKPT}/config.json'))
cfg = PretrainedConfig.from_dict(config_dict)
cfg.text_config = PretrainedConfig.from_dict(config_dict['text_config'])
cfg.name_or_path = CKPT

res = hf_to_mcore_config(cfg)

expect = {
    'hf_model_type': 'qwen4_exp',
    'num_layers': 48,
    'hidden_size': 2560,
    'num_attention_heads': 24,
    'num_query_groups': 2,
    'kv_channels': 256,
    'num_moe_experts': 512,
    'moe_router_topk': 10,
    'moe_ffn_hidden_size': 640,
    'moe_shared_expert_intermediate_size': 640,
    'moe_shared_expert_gate': True,
    'layernorm_zero_centered_gamma': True,
    'attention_output_gate': True,
    'qk_layernorm': True,
    'linear_decoupled_in_proj': True,
    'experimental_attention_variant': 'gated_delta_net',
    'hc_count': 4,
    'hc_lowrank': 320,
    'ple_layer_ids': [2],
    'ple_eos_token_id': 248044,
    'split_ngram_parts': 128,
    'make_ngram_vocab_size_divisible_by': 128,
    'indexer_n_heads': 4,
    'untie_embeddings_and_output_weights': True,
    'partial_rotary_factor': 0.25,
    'rotary_base': 10000000,
}
ok = True
for k, v in expect.items():
    got = res.get(k)
    status = 'OK ' if got == v else 'FAIL'
    if got != v:
        ok = False
    print(f'{status} {k}: {got!r} (expect {v!r})')

la = res.get('linear_attention_freq')
print('linear_attention_freq:', la, type(la))
if isinstance(la, str):
    la_eval = [bool(x) for x in json.loads(la)]
else:
    la_eval = list(la)
expected_pattern = [True, True, True, False] * 12
print('OK  linear pattern' if la_eval == expected_pattern else f'FAIL linear pattern: {la_eval}')
ok &= la_eval == expected_pattern

mf = res.get('moe_layer_freq')
print('moe_layer_freq:', mf)
if isinstance(mf, str):
    mf_eval = [bool(x) for x in json.loads(mf)]
else:
    mf_eval = list(mf)
print('OK  moe pattern all-ones' if mf_eval == [True] * 48 else f'FAIL moe pattern: {mf_eval}')
ok &= mf_eval == [True] * 48

print('rope_scaling:', res.get('rope_scaling'))

# Build the actual ModelConfig to catch __post_init__ validation errors.
# The bridge constructor requires initialized model-parallel groups.
import torch
import torch.distributed as dist
from megatron.core import parallel_state

torch.cuda.set_device(0)
if not dist.is_initialized():
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29555', world_size=1, rank=0)
if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1)

from mcore_bridge.config import ModelConfig

mc = ModelConfig(
    **res,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    context_parallel_size=1,
)
print('ModelConfig OK:', type(mc).__name__, 'seq_parallel:', mc.sequence_parallel)
print('ALL PARSER CHECKS PASSED' if ok else 'PARSER CHECKS FAILED')
sys.exit(0 if ok else 1)
