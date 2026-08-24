# Copyright (c) ModelScope Contributors. All rights reserved.
"""GatedResidualSimple HyperConnection for Qwen3.8-Flash-Next.

This is the low-rank gated multi-stream residual used by Qwen3.8-Flash-Next
(hc_count streams, HC outer / HS inner checkpoint layout). It differs from
the upstream Megatron-LM mHC (mapping_proj + sinkhorn/alpha parameterization,
DSv4 style), so it is implemented here with checkpoint-compatible weight names:

    hc_norm.weight                  (GroupedGemmaRMSNorm, zero-centered)
    input_mix_weight_down.weight    Linear(hc*H, hc_lowrank)
    input_mix_weight_up.weight      Linear(hc_lowrank, hc*H)
    block_inject_weight.weight      Linear(hc*H, hc_count)   [combine only]

All weights are replicated across TP; gradients are reduced across the TP
domain when sequence parallelism is enabled via the ``sequence_parallel``
attribute (same convention as mcore norms).
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple


class GroupedGemmaRMSNorm(nn.Module):
    """Gemma-style zero-centered RMSNorm, optionally grouped along the last dim.

    With ``group_size`` set, variance is computed independently for every
    ``group_size``-wide group (used to normalize each HC stream of size H
    while keeping a distinct affine weight per element of the HC*H layout).
    """

    def __init__(self, hidden_size: int, eps: float, group_size: Optional[int] = None, dtype=None):
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(f'hidden_size ({hidden_size}) must be divisible by group_size ({group_size})')
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped = hidden_states.unflatten(-1, (hidden_states.shape[-1] // self.group_size, self.group_size))
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.variance_epsilon)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class GatedResidualSimple(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``mix`` applies per-stream GemmaRMSNorm then a low-rank sigmoid gate and
    returns the gated mean across streams; ``combine`` injects the block
    output back into each stream through a learned per-stream weight
    ``2 * sigmoid(linear(normed) / hc_count)``.

    Shapes: hyper hidden is ``[..., hc_count * hidden_size]`` with the HC
    stream dimension outer and hidden inner (checkpoint-native layout).
    """

    def __init__(self,
                 hidden_size: int,
                 hc_count: int,
                 hc_lowrank: int,
                 eps: float = 1e-6,
                 use_mix: bool = True,
                 use_combine: bool = True,
                 dtype=None):
        super().__init__()
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hc_count * hidden_size
        # hc_per_branch_norm=True in the reference implementation: normalize
        # each H-sized stream independently with a full HC*H affine weight.
        self.hc_norm = GroupedGemmaRMSNorm(self.hyper_hidden_size, eps=eps, group_size=hidden_size, dtype=dtype)
        if use_mix:
            self.input_mix_weight_down = nn.Linear(self.hyper_hidden_size, hc_lowrank, bias=False, dtype=dtype)
            self.input_mix_weight_up = nn.Linear(hc_lowrank, self.hyper_hidden_size, bias=False, dtype=dtype)
        if use_combine:
            self.block_inject_weight = nn.Linear(self.hyper_hidden_size, hc_count, bias=False, dtype=dtype)
        for param in self.parameters():
            # Replicated across TP; reduce grads across TP when SP is on.
            setattr(param, 'sequence_parallel', True)

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        return self.hc_norm(hyper_input)

    def mix(self, hyper_input: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Mix: RMSNorm -> low-rank gate -> gated mean across streams."""
        assert hyper_input.shape[-1] == self.hyper_hidden_size
        hyper_input_normed = self._normalize(hyper_input)
        # Gate — original mix order: linear+silu then linear+sigmoid.
        gate = F.silu(F.linear(hyper_input_normed, self.input_mix_weight_down.weight) / self.hc_count)
        gate = torch.sigmoid(F.linear(gate, self.input_mix_weight_up.weight)).unflatten(-1, (self.hc_count,
                                                                                               self.hidden_size))
        mixed_input = (gate * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        return mixed_input.to(hyper_input.dtype), (hyper_input, hyper_input_normed)

    def combine(self, block_output: torch.Tensor,
                residuals: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        assert block_output.shape[-1] == self.hidden_size
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hyper_hidden_size
        residual = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        injection_weight = 2.0 * torch.sigmoid(
            F.linear(hyper_input_normed, self.block_inject_weight.weight) / self.hc_count)
        output = residual + block_output.unsqueeze(-2) * injection_weight.unsqueeze(-1)
        return output.flatten(-2).to(hyper_input.dtype)
