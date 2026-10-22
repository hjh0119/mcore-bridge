# Copyright (c) ModelScope Contributors. All rights reserved.
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from mcore_bridge.utils import get_logger

logger = get_logger()


def _cfg_get(config, key: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _get_declared_depth(hf_config, field: Optional[str]) -> Optional[int]:
    if not field:
        return None
    for config in (hf_config, _cfg_get(hf_config, 'text_config')):
        value = _cfg_get(config, field)
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _read_weight_keys(model_dir: Optional[str]) -> Optional[List[str]]:
    if not model_dir or not os.path.isdir(model_dir):
        return None
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index_path = os.path.join(model_dir, name)
        if os.path.isfile(index_path):
            with open(index_path, encoding='utf-8') as f:
                weight_map = json.load(f).get('weight_map', {})
            return list(weight_map)
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))
    if safetensor_files:
        from safetensors import safe_open
        keys = set()
        for path in safetensor_files:
            with safe_open(path, framework='pt') as f:
                keys.update(f.keys())
        return sorted(keys)
    bin_files = sorted(glob.glob(os.path.join(model_dir, '*.bin')))
    if bin_files:
        import torch
        keys = set()
        for path in bin_files:
            state_dict = torch.load(path, map_location='meta', mmap=True, weights_only=True)
            keys.update(state_dict.keys())
        return sorted(keys)
    return None


def _count_standard_sets(keys: List[str], num_layers: int) -> int:
    # Text-only checkpoints use `model.layers.*`, multimodal ones use
    # `model.language_model.layers.*`; both describe the same decoder stack.
    pattern = re.compile(r'^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.')
    indices = set()
    for key in keys:
        match = pattern.match(key)
        if match and int(match.group(1)) >= num_layers:
            indices.add(int(match.group(1)))
    return len(indices)


def _count_prefixed_sets(keys: List[str], prefix: str) -> int:
    name = prefix.split('.')[-1]
    wrapper = r'(?:model\.)?(?:language_model\.)?'
    pattern = re.compile(rf'^{wrapper}{re.escape(name)}\.layers\.(\d+)\.')
    indices = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    if indices:
        return len(indices)
    top_level = re.compile(rf'^{wrapper}{re.escape(name)}\.')
    return 1 if any(top_level.match(key) for key in keys) else 0


def _count_mtp_sets(keys: List[str], prefix: str, hf_config) -> int:
    if prefix in ('model.layers', 'layers'):
        text_config = _cfg_get(hf_config, 'text_config')
        num_layers = _cfg_get(text_config, 'num_hidden_layers') or _cfg_get(hf_config, 'num_hidden_layers')
        if num_layers is None:
            return 0
        return _count_standard_sets(keys, int(num_layers))
    return _count_prefixed_sets(keys, prefix)


def resolve_mtp_num_layers(model_meta, hf_config, model_dir: Optional[str], *, warn: bool = True) -> Optional[int]:
    """Resolve the MTP depth stored in a checkpoint.

    The config's declared depth is only a hint: some checkpoints declare MTP but omit
    the weights (for example MiniCPM-V), so the weight keys decide whether MTP is
    actually available.
    """
    prefix = getattr(model_meta, 'mtp_prefix', None)
    if not prefix:
        return None
    declared = _get_declared_depth(hf_config, getattr(model_meta, 'mtp_depth_field', None))
    keys = _read_weight_keys(model_dir)
    if keys is None:
        if warn:
            logger.warning(f'Could not inspect MTP weights in {model_dir!r}; MTP is disabled. '
                           'Pass `--mtp_num_layers` explicitly if you want to train it.')
        return None
    detected = _count_mtp_sets(keys, prefix, hf_config)
    if detected == 0:
        if declared and warn:
            logger.warning(f'The model config declares {declared} MTP layer(s), but no MTP weights were found '
                           f'under {prefix!r}; MTP is disabled for this checkpoint.')
        return None
    if declared and declared != detected and warn:
        logger.warning(f'The model config declares {declared} MTP layer(s), but the checkpoint stores '
                       f'{detected} physical set(s) under {prefix!r}; using {detected}.')
    return detected


@dataclass(frozen=True)
class MtpResolution:
    num_layers: Optional[int]
    shared_weights: bool
    physical_sets: Optional[int]


def resolve_mtp_config(model_meta, hf_config, model_dir: Optional[str], requested_num_layers=None,
                       requested_shared_weights: Optional[bool] = None) -> MtpResolution:
    """Resolve the logical MTP depth and whether the physical set is reused.

    ``requested_num_layers`` accepts an integer, ``'auto'`` (follow the checkpoint), or
    ``None`` (MTP disabled). When ``requested_shared_weights`` is None, reuse is enabled
    only if the requested depth exceeds the sets actually stored in the checkpoint.
    """
    physical = resolve_mtp_num_layers(model_meta, hf_config, model_dir, warn=False)
    if requested_num_layers == 'auto':
        num_layers = physical
    elif requested_num_layers is None:
        num_layers = None
    else:
        num_layers = int(requested_num_layers)
    if requested_shared_weights is None:
        shared_weights = bool(physical and num_layers and num_layers > physical)
    else:
        shared_weights = bool(requested_shared_weights)
    return MtpResolution(num_layers=num_layers, shared_weights=shared_weights, physical_sets=physical)
