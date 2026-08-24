# Regression test for the ms-swift qwen4_exp transformers-compat shim.
#
# `_compat_qwen4_exp_transformers()` (swift/model/models/qwen.py) polyfills a
# handful of helpers that only exist in newer transformers, so that the bundled
# qwen4_exp modeling module can be imported on the installed version. Those
# polyfills are only reached on the multimodal (vision) path, which meant a
# `NameError` in `get_vision_attention_seqlens` (returning an undefined
# `max_seqlens` instead of `max_seqlen`) went unnoticed by the text-only tests.
#
# This test calls each polyfill directly with realistic inputs so any
# NameError / signature drift fails fast, without needing a GPU or checkpoint.
#
# Run:
#   python tests/test_qwen4_exp_compat_shim.py
import sys

MS_SWIFT = '/mnt/workspace/hjh/workspace/ms-swift'


def main():
    import torch
    sys.path.insert(0, MS_SWIFT)
    from swift.model.models.qwen import _compat_qwen4_exp_transformers
    assert _compat_qwen4_exp_transformers(), 'ms-swift qwen4_exp compat failed'

    from transformers import vision_utils
    from transformers.utils import generic as hf_generic

    failures = []

    def check(name, fn):
        try:
            fn()
            print(f'OK   {name}')
        except Exception as exc:  # noqa: BLE001 - report every polyfill failure
            failures.append(f'{name}: {type(exc).__name__}: {exc}')
            print(f'FAIL {name}: {type(exc).__name__}: {exc}')

    # grid_thw: one 2x4x4 video-ish grid (t, h, w), spatial_merge_size=2.
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)

    def _seqlens():
        # The bug: this returned an undefined `max_seqlens`.
        cu, max_seqlen = vision_utils.get_vision_attention_seqlens(
            grid_thw, config=None, merge_temporal=False, kwargs=None)
        assert cu is not None and cu.numel() >= 2, f'bad cu_seqlens: {cu}'
        # max_seqlen is None unless flash-attention is requested; both are fine,
        # what matters is that the call does not raise.
        assert max_seqlen is None or int(max_seqlen) > 0, f'bad max_seqlen: {max_seqlen}'
        # merge_temporal=True (clip-level attention) must also be accepted: the
        # installed get_vision_cu_seqlens has no such parameter, so the shim has
        # to provide it. Multi-frame grid so the two modes actually differ.
        multi = torch.tensor([[2, 4, 4]], dtype=torch.long)
        cu_frame, _ = vision_utils.get_vision_attention_seqlens(
            multi, config=None, merge_temporal=False, kwargs=None)
        cu_clip, _ = vision_utils.get_vision_attention_seqlens(
            multi, config=None, merge_temporal=True, kwargs=None)
        # per-frame: 2 segments of h*w=16 -> [0,16,32]; clip: one 32 -> [0,32]
        assert cu_frame.tolist() == [0, 16, 32], f'per-frame cu_seqlens: {cu_frame.tolist()}'
        assert cu_clip.tolist() == [0, 32], f'clip cu_seqlens: {cu_clip.tolist()}'

    def _interp_taps():
        index = torch.arange(4, dtype=torch.long)
        size = torch.full((4, ), 4, dtype=torch.long)
        taps, weights = vision_utils._interpolation_axis_taps_weights(
            index, size, 8, 'bilinear', False, 'border')
        assert taps.shape == weights.shape, f'{taps.shape} vs {weights.shape}'
        assert taps.min() >= 0 and taps.max() <= 7, f'taps out of range: {taps}'

    def _interp_indices():
        indices, weights = vision_utils.get_vision_interpolation_indices_and_weights(
            grid_thw, num_grid_per_side=8, mode='bilinear', align_corners=False,
            spatial_merge_size=2, padding='border', kwargs=None)
        n_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
        assert indices.shape[0] == n_tokens, f'{indices.shape[0]} != {n_tokens}'
        assert indices.shape == weights.shape, f'{indices.shape} vs {weights.shape}'
        assert indices.min() >= 0 and indices.max() < 8 * 8, f'indices out of range: {indices.max()}'

    def _max_seqlen():
        cu = torch.tensor([0, 5, 12], dtype=torch.long)
        out = hf_generic.get_max_seqlen(cu, config=None, kwargs={'max_seqlen': 7})
        assert out == 7, f'explicit kwargs max_seqlen ignored: {out}'

    def _causal_mask():
        # allow_is_causal_skip=False must yield a materialized 4D bool mask,
        # which the QSA indexer relies on.
        from transformers import masking_utils as hf_masking
        import inspect
        params = inspect.signature(hf_masking.create_causal_mask).parameters
        assert 'config' in params, f'unexpected create_causal_mask signature: {list(params)}'

    check('get_vision_attention_seqlens', _seqlens)
    check('_interpolation_axis_taps_weights', _interp_taps)
    check('get_vision_interpolation_indices_and_weights', _interp_indices)
    check('get_max_seqlen', _max_seqlen)
    check('create_causal_mask signature', _causal_mask)

    print()
    if failures:
        print(f'COMPAT SHIM TEST FAILED ({len(failures)})')
        for f in failures:
            print(' ', f)
        return 1
    print('COMPAT SHIM TEST PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
