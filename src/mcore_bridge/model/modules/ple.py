# Copyright (c) ModelScope Contributors. All rights reserved.
"""Position-Learning Enhancement (PLE) for Qwen3.8-Flash-Next.

Training-time implementation of the n-gram hash embedding + sign-sqrt gated
lookup + dilated short conv used by the internal vLLM implementation
(`vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`):

  tokens -> trigram XOR hash -> ngram_embedding lookup
         -> key_proj/value_proj -> sign-sqrt gate (k.q/sqrt(H))
         -> grouped norms -> dilated causal short-conv (silu)
         -> gated_value + conv_output   (added onto the multi-stream hidden)

Differences vs. the inference implementation (training-only, by design):
  - No per-request conv state cache: the conv is recomputed over the whole
    packed sequence with zero left context at each sample start.
  - ``ngram_context`` (cross-forward token history) does not exist during
    training; every sample starts fresh (eos-padded context), matching a
    fresh inference request.
  - Sequence parallel: PLE runs on each SP rank's local shard; the first
    ``ngram_size-1``/conv-history tokens of a shard lose cross-shard context.
"""
import math
import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel import VocabParallelEmbedding
from torch import nn
from typing import List, Optional

from .hyper_connection_gated import GroupedGemmaRMSNorm

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class NGramHashEmbedding(nn.Module):
    """N-gram XOR-hash embedding table (vocab-sharded across TP).

    Hash ids: for an n-gram of size ``n`` at position t,
        id_h = ((t_t*m_0) ^ (t_{t-1}*m_1) ^ ... ^ (t_{t-n+1}*m_{n-1})) mod p_h
    where multipliers are derived from splitmix64(seed + 10007*layer_id) and
    p_h is the (layer*ngram_heads + h)-th prime after ngram_vocab_size_base.
    Each head owns its own prime-sized slice of one padded flat table.
    """

    def __init__(
        self,
        embedding_dim: int,
        ngram_size: int,
        heads_per_ngram: int,
        ngram_vocab_size_base: int,
        make_vocab_divisible_by: int,
        ple_dense_layer_id: int,
        vocab_size: int,
        eos_token_id: int,
        seed: int = 1234,
        config=None,
    ):
        super().__init__()
        assert ngram_size >= 2
        self.ngram_size = int(ngram_size)
        self.heads_per_ngram = int(heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        assert embedding_dim % self.ngram_heads == 0
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(eos_token_id)
        # checkpoint shards the table into split_ngram_parts row blocks
        self.split_ngram_parts = int(getattr(config, 'split_ngram_parts', None) or 128)

        # Multipliers (splitmix64 derived, checkpoint-persistent).
        max_multiplier = ((1 << 63) - 1) // int(vocab_size)
        half_bound = max(1, max_multiplier // 2)
        base_seed = int(seed) + _PLE_LAYER_PRIME * int(ple_dense_layer_id)
        multipliers = []
        for index in range(self.ngram_size):
            value = base_seed + _SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
        self.register_buffer('layer_multipliers', torch.tensor(multipliers, dtype=torch.long), persistent=True)

        # Per-head prime table sizes/offsets (checkpoint-persistent).
        sizes: List[int] = []
        offsets: List[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = int(ple_dense_layer_id) * self.ngram_heads + local_head
            size = _nth_prime_after(int(ngram_vocab_size_base) - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.register_buffer('ngram_heads_vocab_sizes', torch.tensor(sizes, dtype=torch.long), persistent=True)
        self.register_buffer('ngram_heads_offsets', torch.tensor(offsets, dtype=torch.long), persistent=True)
        divisor = int(make_vocab_divisible_by)
        self.org_vocab_size = offset
        padded_vocab_size = ((offset + divisor - 1) // divisor) * divisor
        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            init_method=torch.nn.init.normal_,
            config=config,
        )

    @staticmethod
    def _shift_precompute(tokens: torch.Tensor, eos_token_id: int):
        # tokens: [rows, L]; returns (positions, position_in_segment) where
        # segments are reset after every eos token (request boundary logic).
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(tokens, positions, position_in_segment, shift, eos_token_id):
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [rows, L] int64 -> embeddings [rows, L, embedding_dim]."""
        tokens = tokens.long()
        context_len = self.ngram_size - 1
        context_prefix = tokens.new_full((tokens.shape[0], context_len), self.eos_token_id)
        context = torch.cat([context_prefix, tokens], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(context, self.eos_token_id)
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(context, positions_2d, position_in_segment, shift, self.eos_token_id))
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[index] * self.layer_multipliers[index])
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            # drop the eos-context prefix columns
            id_blocks.append(ids[:, context_len:])
        ngram_ids = torch.cat(id_blocks, dim=-1)  # [rows, L, ngram_heads]
        return self.ngram_embedding(ngram_ids).flatten(-2)


class PLELayer(nn.Module):
    """PLE branch attached to selected decoder layers.

    Consumes raw input tokens and the multi-stream hidden ``[..., hc*H]`` and
    returns an additive term of width ``hc*H``. Checkpoint names under the
    layer prefix ``ple.``:
        ple_embedding.{layer_multipliers,ngram_heads_offsets,ngram_heads_vocab_sizes}
        ple_embedding.ngram_embedding.shard_{i}.weight
        key_proj.weight / value_proj.weight
        norm_key.weight / norm_query.weight / norm_conv.weight
        conv1d.weight
    """

    def __init__(self, config, ple_dense_layer_id: int):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        ple_embed_dim = int(config.ple_embed_dim)
        self.ple_embedding = NGramHashEmbedding(
            embedding_dim=ple_embed_dim,
            ngram_size=config.ngram_size,
            heads_per_ngram=config.heads_per_ngram,
            ngram_vocab_size_base=config.ngram_vocab_size_base,
            make_vocab_divisible_by=config.make_ngram_vocab_size_divisible_by,
            ple_dense_layer_id=ple_dense_layer_id,
            vocab_size=config.padded_vocab_size,
            eos_token_id=getattr(config, 'ple_eos_token_id', 0),
            seed=getattr(config, 'ple_seed', 1234),
            config=config,
        )
        # Replicated projections (reference uses ReplicatedLinear).
        self.key_proj = nn.Linear(ple_embed_dim, self.hc_hidden_size, bias=False, dtype=config.params_dtype)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False, dtype=config.params_dtype)
        norm_args = (self.hc_hidden_size, config.layernorm_epsilon, self.hidden_size)
        self.norm_key = GroupedGemmaRMSNorm(*norm_args, dtype=config.params_dtype)
        self.norm_query = GroupedGemmaRMSNorm(*norm_args, dtype=config.params_dtype)
        self.norm_conv = GroupedGemmaRMSNorm(*norm_args, dtype=config.params_dtype)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=(self.conv_kernel_size - 1) * self.short_conv_dilation,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=config.params_dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        for name, param in self.named_parameters():
            if name.startswith(('key_proj', 'value_proj')) or name.endswith('norm.weight') or name.startswith(
                ('norm_key', 'norm_query', 'norm_conv')) or name.startswith('conv1d'):
                # Replicated across TP; reduce grads across TP when SP is on.
                setattr(param, 'sequence_parallel', True)

    def _apply_norm(self, norm, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def _short_conv(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [rows, L, C]; per-row dilated causal conv with zero history
        # at the row start (fresh request semantics).
        x = inputs.transpose(1, 2)  # [rows, C, L]
        out = F.conv1d(x, self.conv1d.weight, groups=self.hc_hidden_size, dilation=self.short_conv_dilation,
                       padding=(self.conv_kernel_size - 1) * self.short_conv_dilation)[..., :x.size(-1)]
        return F.silu(out).transpose(1, 2)

    def compute(self, hidden_states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """hidden_states/tokens: [rows, L, nH]/[rows, L] -> additive [rows, L, nH]."""
        embeddings = self.ple_embedding(tokens)
        key = self.key_proj(embeddings)
        value = self.value_proj(embeddings)
        rows, seq_len = hidden_states.shape[:2]
        key = key.reshape(rows, seq_len, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(rows, seq_len, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output = self._short_conv(normalized)
        return gated_value.flatten(-2) + conv_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """hidden_states: [s, b, nH] (bsh) or thd [T, 1, nH]; input_ids: [b, s] or [1, T]."""
        thd = packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', 'bshd') == 'thd'
        if thd:
            num_samples = packed_seq_params.num_samples
            max_len = packed_seq_params.max_seqlen_q.item()
            cu = packed_seq_params.cu_seqlens_q
            total = hidden_states.shape[0]
            hid = hidden_states.new_zeros((num_samples, max_len, hidden_states.shape[-1]))
            toks = input_ids.new_full((num_samples, max_len), self.ple_embedding.eos_token_id)
            for i in range(num_samples):
                start, end = int(cu[i]), int(cu[i + 1])
                hid[i, :end - start] = hidden_states[start:end, 0]
                toks[i, :end - start] = input_ids[0, start:end]
            res = self.compute(hid, toks)
            out = res.new_zeros((total, 1, res.shape[-1]))
            for i in range(num_samples):
                start, end = int(cu[i]), int(cu[i + 1])
                out[start:end, 0] = res[i, :end - start]
            return out
        else:
            # [s, b, nH] -> [b, s, nH]; input_ids [b, s]
            hid = hidden_states.transpose(0, 1)
            res = self.compute(hid, input_ids)
            return res.transpose(0, 1).contiguous()
