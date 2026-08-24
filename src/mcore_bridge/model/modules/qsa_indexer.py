# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight holder for the Qwen3.8-Flash-Next QSA indexer.

The QSA two-stage sparse selection (group scoring -> token expansion) is an
inference-time efficiency mechanism built on paged KV caches. During
Megatron training the attention of QSA layers runs as dense causal attention
with identical qkv/o weights; the indexer weights are loaded/saved (and stay
trainable) but do not participate in the training forward pass.

Checkpoint names under ``self_attn.indexer.``:
    index_qk_proj.weight   Linear(H, (indexer_n_heads + indexer_kv_heads) * indexer_head_dim)
    q_layernorm.weight     GemmaRMSNorm(indexer_head_dim)  (zero-centered)
    k_layernorm.weight     GemmaRMSNorm(indexer_head_dim)  (zero-centered)
"""
import torch
from torch import nn


class _ZeroCenteredRMSNorm(nn.Module):
    """Gemma-style zero-centered RMSNorm ((1 + weight) scaling)."""

    def __init__(self, hidden_size: int, eps: float, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        output = hidden_states * torch.rsqrt(variance + self.eps)
        return (output * (1.0 + self.weight.float())).to(input_dtype)


class QSAIndexerWeights(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        index_n_heads = int(config.indexer_n_heads)
        index_kv_heads = int(config.indexer_kv_heads)
        index_head_dim = int(config.indexer_head_dim)
        # Replicated projection (reference uses ReplicatedLinear).
        self.index_qk_proj = nn.Linear(
            config.hidden_size, (index_n_heads + index_kv_heads) * index_head_dim,
            bias=False,
            dtype=config.params_dtype)
        self.q_layernorm = _ZeroCenteredRMSNorm(index_head_dim, eps=config.layernorm_epsilon,
                                                dtype=config.params_dtype)
        self.k_layernorm = _ZeroCenteredRMSNorm(index_head_dim, eps=config.layernorm_epsilon,
                                                dtype=config.params_dtype)
        setattr(self.index_qk_proj.weight, 'sequence_parallel', config.sequence_parallel)

    def forward(self, *args, **kwargs):
        raise RuntimeError('QSAIndexerWeights is a checkpoint holder; the QSA selection '
                           'runs only in the inference engine (dense attention is used in training).')
