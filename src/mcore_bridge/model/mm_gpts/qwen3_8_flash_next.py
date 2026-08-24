# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen3.8-Flash-Next multimodal registration.

Vision tower: Qwen3-VL style (Qwen3VLVisionConfig based, deepstack enabled),
so ``Qwen3VL_Vit`` is reused. The deepstack injection differs from qwen3_vl:
decoder hidden states are multi-stream ``[s, b, hc*H]`` (HC outer / HS inner),
so each deepstack level embedding is broadcast to every HC stream before the
add, matching the internal vLLM implementation.

Text-only checkpoints load through the same meta with
``language_model_only=True`` (requires an empty ``deepstack_visual_indexes``).
"""
import torch

from mcore_bridge.utils import get_env_args

from ..constant import ModelType
from ..gpts.qwen3_8_flash_next import (Qwen3_8FlashNextBridge, Qwen3_8FlashNextLoader,
                                       Qwen38FlashNextTransformerBlock)
from ..register import ModelMeta, register_model
from .qwen3_vl import Qwen3VL_Vit


class Qwen3_8FlashNextVit(Qwen3VL_Vit):

    def prepare_model(self, hf_config):
        from transformers.models.qwen3_vl import Qwen3VLVisionModel as VisionModel
        self.visual = VisionModel._from_config(hf_config.vision_config)


class Qwen3_8FlashNextTransformerBlock(Qwen38FlashNextTransformerBlock):
    """Deepstack-aware block for multi-stream (HC) hidden states."""

    def _layer_forward(self, layer, hidden_states, **kwargs):
        deepstack_visual_embeds = kwargs.pop('deepstack_visual_embeds', None)
        visual_pos_masks = kwargs.pop('visual_pos_masks', None)
        hidden_states, context = super()._layer_forward(layer, hidden_states, **kwargs)
        layer_number = layer.layer_number - 1
        if deepstack_visual_embeds is not None and layer_number in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_number],
            )
        return hidden_states, context

    def forward(self, *args, **kwargs):
        deepstack_visual_embeds = kwargs.get('deepstack_visual_embeds')
        if deepstack_visual_embeds is not None:
            assert len(deepstack_visual_embeds) <= len(
                self.layers), (f'len(deepstack_visual_embeds): {len(deepstack_visual_embeds)}, '
                               f'len(self.layers): {len(self.layers)}.')
        return super().forward(*args, **kwargs)

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor):
        hc_count = getattr(self.config, 'hc_count', 1) or 1
        if visual_pos_masks is None:
            return hidden_states + visual_embeds.mean() * 0
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        if hc_count > 1:
            # hidden states are [s, b, hc*H] (HC outer, HS inner): broadcast
            # the deepstack embedding to every stream before the add.
            visual_embeds = visual_embeds.unsqueeze(-2).expand(
                *visual_embeds.shape[:-1], hc_count, self.config.hidden_size).flatten(-2)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class Qwen3_8FlashNextMMBridge(Qwen3_8FlashNextBridge):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_mixer_prefix = 'model.language_model.'


class Qwen3_8FlashNextMMLoader(Qwen3_8FlashNextLoader):
    transformer_block = Qwen3_8FlashNextTransformerBlock


use_mcore_gdn = get_env_args('USE_MCORE_GDN', bool, True)

if use_mcore_gdn:
    register_model(
        ModelMeta(
            ModelType.qwen3_8_flash_next,
            ['qwen3_8_flash_next', 'qwen4_exp'],
            bridge_cls=Qwen3_8FlashNextMMBridge,
            visual_cls=Qwen3_8FlashNextVit,
            loader=Qwen3_8FlashNextMMLoader,
        ))
