"""L'empilement par bandes de lignes doit donner exactement le même résultat qu'une seule bande."""
import numpy as np
import cv2

from astrostack.stacker import stack_frames, feather
from astrostack.stars import warp_to_ref_valid, warp_to_ref_valid_band


def _scene(h=70, w=60, n=5, seed=0):
    rng = np.random.default_rng(seed)
    frames = [(rng.random((h, w, 3)) * 60000).astype(np.float32) for _ in range(n)]
    transforms = [None]
    transforms.append(np.array([[1, 0, 2.5], [0, 1, -1.5]], np.float64))            # affine
    transforms.append(np.array([[1.001, 0.002, 1.2], [-0.002, 1.0, -0.8], [1e-6, 0, 1]]))  # homographie
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    transforms.append(("remap", mx + 0.7 * np.sin(my / 9), my + 0.4))               # remap TPS-like
    transforms.append(np.array([[1, 0, -3.0], [0, 1, 4.0], [0, 0, 1]]))
    fg = np.zeros((h, w), np.float32)
    fg[:20, 15:40] = 1.0
    fg = feather(np.where(fg > 0, 255, 0).astype(np.uint8), sigma=2.0)
    return frames, transforms[:n], fg


def test_band_warp_matches_full_warp():
    frames, transforms, _ = _scene()
    h, w = frames[0].shape[:2]
    for M in transforms:
        full, vfull = warp_to_ref_valid(frames[1], M, (w, h))
        for r0, r1 in [(0, 17), (17, 50), (50, h)]:
            band, vband = warp_to_ref_valid_band(frames[1], M, (w, h), r0, r1)
            # OpenCV quantifie l'interpolation (1/32 px) : écart max < 1 niveau sur 65535
            assert np.abs(band - full[r0:r1]).max() < 1.0
            assert np.array_equal(vband, vfull[r0:r1])


def test_sigma_bands_equal_single_band():
    frames, transforms, fg = _scene()
    get = lambda i: frames[i]
    one = stack_frames(get, transforms, fg, mode="sigma", band_bytes=10**12)
    banded = stack_frames(get, transforms, fg, mode="sigma", band_bytes=60 * 3 * 4 * 12 * 9)
    assert np.allclose(one, banded, atol=1e-2)


def test_robust_bands_equal_single_band():
    frames, transforms, fg = _scene()
    get = lambda i: frames[i]
    one = stack_frames(get, transforms, fg, mode="robust", band_bytes=10**12)
    banded = stack_frames(get, transforms, fg, mode="robust", band_bytes=60 * 3 * 4 * 12 * 9)
    assert np.allclose(one, banded, atol=1e-2)


def test_progress_reports_band_tags_only_when_banded():
    frames, transforms, fg = _scene()
    msgs = []
    stack_frames(lambda i: frames[i], transforms, fg, progress_cb=lambda s, i, n, m: msgs.append(m),
                 band_bytes=60 * 3 * 4 * 12 * 9)
    assert any("[bande 1/" in m for m in msgs)
    msgs.clear()
    stack_frames(lambda i: frames[i], transforms, fg, progress_cb=lambda s, i, n, m: msgs.append(m),
                 band_bytes=10**12)
    assert msgs and not any("[bande" in m for m in msgs)


def test_fast_sigma_close_to_precise_and_bands_consistent():
    frames, transforms, fg = _scene()
    get = lambda i: frames[i]
    precise = stack_frames(get, transforms, fg, mode="sigma", band_bytes=10**12)
    fast_one = stack_frames(get, transforms, fg, mode="sigma", band_bytes=10**12, fast=True)
    fast_banded = stack_frames(get, transforms, fg, mode="sigma", band_bytes=60 * 3 * 4 * 12 * 9, fast=True)
    assert np.allclose(precise, fast_one, atol=1e-2)
    assert np.allclose(fast_one, fast_banded, atol=1e-2)


def test_fast_sigma_reuses_warps_instead_of_reloading_frames():
    frames, transforms, fg = _scene()
    calls = {"precise": 0, "fast": 0}

    def counting(key):
        def get(i):
            calls[key] += 1
            return frames[i]
        return get

    stack_frames(counting("precise"), transforms, fg, mode="sigma", band_bytes=10**12)
    stack_frames(counting("fast"), transforms, fg, mode="sigma", band_bytes=10**12, fast=True)
    assert calls["fast"] < calls["precise"]
