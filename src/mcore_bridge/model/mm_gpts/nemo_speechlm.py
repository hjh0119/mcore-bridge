# Copyright (c) ModelScope Contributors. All rights reserved.
"""NeMo SpeechLM with a replicated audio tower and a Nemotron-H backbone."""
import os
import re
import torch
from dataclasses import dataclass
from functools import partial
from transformers import PretrainedConfig
from types import SimpleNamespace
from typing import Optional

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils.safetensors import LazyTensor

from ..constant import ModelType
from ..gpts.nemotron_h import _HYBRID_MODEL_AVAILABLE, NemotronHBridge, NemotronHLoader
from ..mm_gpt_model import MultimodalGPTModel
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


def _load_nemo_expert(weight, expert_idx):
    return weight.load_slice((expert_idx, )).transpose(0, 1)


def _pad_runtime_vocab(weight, target_size):
    tensor = weight.load()
    if tensor.shape[0] < target_size:
        padding = tensor.new_zeros((target_size - tensor.shape[0], *tensor.shape[1:]))
        tensor = torch.cat([tensor, padding])
    return tensor


@dataclass
class NemoSpeechLMConfig(ModelConfig):
    make_vocab_size_divisible_by: int = 128
    actual_vocab_size: Optional[int] = None

    def __post_init__(self):
        runtime_vocab_size = self.padded_vocab_size
        output_vocab_size = self.actual_vocab_size
        divisor = self.make_vocab_size_divisible_by * self.tensor_model_parallel_size
        self.padded_vocab_size = ((runtime_vocab_size + divisor - 1) // divisor) * divisor
        super().__post_init__()
        self.actual_vocab_size = runtime_vocab_size if output_vocab_size is None else output_vocab_size


class NemoSpeechLMBridge(NemotronHBridge):
    hf_embed_key = 'llm.model.embed_tokens.weight'
    hf_layers_prefix = 'llm.model.layers'
    hf_final_layernorm_key = 'llm.model.norm.weight'
    hf_lm_head_key = 'llm.lm_head.weight'
    hf_mtp_prefix = 'llm.mtp.layers'
    hf_state_dict_mapping = {
        'mixer._fp32_params.A_log': 'mixer.A_log',
        'mixer._fp32_params.D': 'mixer.D',
        'mixer._fp32_params.dt_bias': 'mixer.dt_bias',
    }

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        result = {}
        for key, value in hf_state_dict.items():
            for old_key, new_key in self.hf_state_dict_mapping.items():
                if not to_mcore:
                    old_key, new_key = new_key, old_key
                if old_key in key:
                    key = key.replace(old_key, new_key)
                    break
            result[key] = value
        return result

    def _set_state_dict(self, mg_module, mg_key, hf_state_dict, hf_key, to_mcore, *, offset=0, is_expert=False):
        if to_mcore and not self._peft_format and hf_key in {self.hf_embed_key, self.hf_lm_head_key}:
            hf_state_dict = hf_state_dict.copy()
            hf_state_dict[hf_key] = LazyTensor(
                loader=partial(_pad_runtime_vocab, hf_state_dict[hf_key], self.config.padded_vocab_size))
        return super()._set_state_dict(
            mg_module, mg_key, hf_state_dict, hf_key, to_mcore, offset=offset, is_expert=is_expert)

    def _set_mlp_state(self,
                       mg_mlp,
                       hf_state_dict,
                       hf_prefix: str,
                       layer_idx: int,
                       to_mcore: bool,
                       ep_rank=None,
                       is_mtp: bool = False):
        if not to_mcore or ep_rank is None or self._peft_format:
            return super()._set_mlp_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, ep_rank, is_mtp)
        hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        num_local_experts = self.config.num_moe_experts // self.ep_size
        start_idx = ep_rank * num_local_experts
        for source, target in [('gate_and_up_projs', 'up_proj'), ('down_projs', 'down_proj')]:
            weight = hf_state_dict.pop(f'{source}.weight', None)
            if weight is None:
                weight = hf_state_dict.pop(source)
            for local_idx in range(num_local_experts):
                expert_idx = start_idx + local_idx
                hf_state_dict[f'{expert_idx}.{target}.weight'] = LazyTensor(
                    loader=partial(_load_nemo_expert, weight, expert_idx))
        return super()._set_mlp_state(mg_mlp, hf_state_dict, '', layer_idx, True, ep_rank, is_mtp)

    def _pack_nemo_experts(self, hf_state_dict):
        for hf_name, nemo_name in [('up_proj', 'gate_and_up_projs'), ('down_proj', 'down_projs')]:
            pattern = re.compile(rf'^(.*)\.(\d+)\.{hf_name}\.weight$')
            weights = {}
            for key in list(hf_state_dict):
                match = pattern.match(key)
                if match:
                    weights[int(match.group(2))] = hf_state_dict.pop(key)
                    prefix = match.group(1)
            if weights:
                assert len(weights) == self.config.num_moe_experts
                if prefix.endswith('.experts'):
                    prefix = prefix[:-len('.experts')]
                hf_state_dict[f'{prefix}.experts.{nemo_name}'] = torch.stack(
                    [weights[i].transpose(0, 1) for i in range(self.config.num_moe_experts)])
        return hf_state_dict

    def _set_moe_state(self, mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, is_mtp=False):
        result = super()._set_moe_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore, is_mtp)
        if not to_mcore and not self._peft_format:
            result = self._pack_nemo_experts(result)
        return result

    def _set_module(self, mg_module, hf_state_dict, hf_prefix, to_mcore):
        if self._peft_format or not to_mcore or hf_prefix != 'perception.':
            return super()._set_module(mg_module, hf_state_dict, hf_prefix, to_mcore)
        if mg_module is None:
            return {}
        state = {
            key: value.load()
            for key, value in self._remove_prefix(hf_state_dict, hf_prefix).items() if not key.endswith('._extra_state')
        }
        incompatible = mg_module.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.endswith('._extra_state')]
        unexpected = [key for key in incompatible.unexpected_keys if not key.endswith('._extra_state')]
        assert not missing and not unexpected, (
            f'Perception checkpoint mismatch: missing={missing}, unexpected={unexpected}')
        return {}

    def get_hf_meta_model(self):
        return SimpleNamespace(_auto_class=None)

    def save_weights(self, mg_models, output_dir, peft_format=False, save_missing_weights=False, **kwargs):
        if not peft_format:
            source_model_dir = save_missing_weights if isinstance(save_missing_weights, str) else self._source_model_dir
            if source_model_dir is None or not os.path.isdir(source_model_dir):
                raise ValueError('Exporting a complete NeMo SpeechLM checkpoint requires the source checkpoint to '
                                 'preserve MTP, CTC and _extra_state weights. Pass the original --model directory.')
            save_missing_weights = save_missing_weights or True
        return super().save_weights(
            mg_models, output_dir, peft_format=peft_format, save_missing_weights=save_missing_weights, **kwargs)


class NemoSpeechLMVit(HuggingFaceVit):
    module_mapping = {'perception': 'perception'}
    _vision_tower = ['perception.encoder']
    _aligner = ['perception.proj']
    test_mm_type = 'audio'

    def prepare_model(self, hf_config: PretrainedConfig):
        from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT
        from nemo.collections.speechlm2.modules import AudioPerceptionModule
        from omegaconf import DictConfig

        self.perception = AudioPerceptionModule(DictConfig(hf_config.perception))
        self.perception.encoder = ParallelExpertEncoderPT.from_inline_config(
            hf_config.pe_encoder_config, map_location='cpu')
        self.perception.encoder.chunk_size_seconds = hf_config.encoder_chunk_size_seconds
        self.perception.encoder.online_inference_enabled = False
        self.perception.encoder.spk_kernel_scale = hf_config.spk_kernel_scale
        self.perception.preprocessor.featurizer.normalize = None
        self.perception.preprocessor.requires_grad_(False)
        self.audio_token_id = hf_config.audio_token_id

    def _encode_audio(self, audio_signal, audio_signal_length, device):
        if isinstance(audio_signal, (list, tuple)):
            max_length = max(audio.shape[-1] for audio in audio_signal)
            audio_signal = torch.stack(
                [torch.nn.functional.pad(audio, (0, max_length - audio.shape[-1])) for audio in audio_signal])
        audio_signal = audio_signal.to(device=device, dtype=torch.float32)
        audio_signal_length = audio_signal_length.to(device=device)
        embeddings, lengths = self.perception(input_signal=audio_signal, input_signal_length=audio_signal_length)
        return torch.cat([embedding[:length] for embedding, length in zip(embeddings, lengths)])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        audio_signal = kwargs.get('audio_signal')
        is_dummy = audio_signal is None
        if is_dummy:
            audio_signal = inputs_embeds.new_zeros((1, 16000), dtype=torch.float32)
            audio_signal_length = input_ids.new_tensor([audio_signal.shape[-1]])
        else:
            audio_signal_length = kwargs['audio_signal_length']
        audio_embeds = self._encode_audio(audio_signal, audio_signal_length, inputs_embeds.device)
        if is_dummy:
            return inputs_embeds + audio_embeds.mean().to(inputs_embeds.dtype) * 0.
        audio_mask = input_ids == self.audio_token_id
        expected = (int(audio_mask.sum()), inputs_embeds.shape[-1])
        if audio_embeds.shape != expected:
            raise ValueError(f'Audio embeddings {tuple(audio_embeds.shape)} do not match placeholders {expected}.')
        audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(audio_mask.unsqueeze(-1).expand_as(inputs_embeds), audio_embeds)


if _HYBRID_MODEL_AVAILABLE:
    from ..hybrid_model import HybridModel

    class NemoSpeechLMGPTModel(MultimodalGPTModel):
        language_model_cls = HybridModel

        def __init__(self, config, *args, **kwargs):
            bridge = config.bridge
            config.bridge = None
            try:
                super().__init__(config, *args, **kwargs)
            finally:
                config.bridge = bridge


class NemoSpeechLMLoader(NemotronHLoader):
    model_cls = NemoSpeechLMGPTModel if _HYBRID_MODEL_AVAILABLE else None

    def __init__(self, config):
        if not _HYBRID_MODEL_AVAILABLE:
            raise ImportError('NeMo SpeechLM requires megatron-core with models.hybrid support.')
        if config.context_parallel_size > 1:
            raise NotImplementedError('NeMo SpeechLM requires --context_parallel_size 1.')
        if config.mtp_num_layers and not getattr(config, 'mtp_use_repeated_layer', False):
            raise NotImplementedError('NeMo SpeechLM MTP has one physical repeated pattern; '
                                      'use --mtp_shared_weights true or --mtp_num_layers 1.')
        if config.fp8 or getattr(config, 'fp4', None):
            raise NotImplementedError('NeMo SpeechLM FP8/FP4 training is not supported.')
        if getattr(config, 'padding_free', False) or getattr(config, 'packing', False):
            raise NotImplementedError('NeMo SpeechLM requires --padding_free false --packing false.')
        super().__init__(config)


if _HYBRID_MODEL_AVAILABLE:
    register_model(
        ModelMeta(
            ModelType.nemo_speechlm, ['nemo_speechlm'],
            bridge_cls=NemoSpeechLMBridge,
            visual_cls=NemoSpeechLMVit,
            loader=NemoSpeechLMLoader,
            config_cls=NemoSpeechLMConfig,
            mtp_prefix='llm.mtp.layers',
            mtp_depth_field='mtp.num_nextn_predict_layers',
            mtp_repeated_field='mtp.use_repeated_layer',
            mtp_layers_per_depth=2))
