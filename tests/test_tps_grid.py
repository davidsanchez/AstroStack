import numpy as np

from astrostack import stars


def _synthetic_matches(n, h_s, w_s, seed=0):
    rng = np.random.default_rng(seed)
    src = np.column_stack([rng.uniform(0, w_s, n), rng.uniform(0, h_s, n)])
    warp = np.column_stack([np.sin(src[:, 1] / 200) * 3, np.cos(src[:, 0] / 300) * 2])
    dst = src + rng.normal(0, 0.8, src.shape) + warp
    return src, dst


def test_coarse_tps_grid_matches_exact_within_tolerance():
    # Pas 4 (défaut) : écart max ~0,27 px sur ce cas dense (200 étoiles pour 107 000 px) ;
    # pas 2 donnerait ~0,08 px mais coûte ~4x plus cher. Tolérance non-régression, pas une cible de qualité.
    h_s, w_s = 267, 400
    s = 400 / 1500
    full_wh = (1500, 1000)
    src, dst = _synthetic_matches(200, h_s, w_s)
    mx_ref, my_ref = stars._local_tps_maps((h_s, w_s), src, dst, s, full_wh, grid_step=1)
    mx, my = stars._local_tps_maps((h_s, w_s), src, dst, s, full_wh)
    assert mx.shape == mx_ref.shape and my.shape == my_ref.shape
    err = np.maximum(np.abs(mx - mx_ref), np.abs(my - my_ref))
    assert err.max() < 0.35
    assert err.mean() < 0.03


def test_upsample_aligned_keeps_corners_and_linear_fields():
    # Un champ linéaire doit être reproduit exactement, coins compris.
    h_c, w_c = 5, 7
    yy, xx = np.meshgrid(np.arange(h_c, dtype=np.float64), np.arange(w_c, dtype=np.float64), indexing="ij")
    a = 2.0 * xx - 3.0 * yy + 1.0
    out = stars._upsample_aligned(a, 21, 31)
    gy, gx = np.meshgrid(np.linspace(0, h_c - 1, 21), np.linspace(0, w_c - 1, 31), indexing="ij")
    np.testing.assert_allclose(out, 2.0 * gx - 3.0 * gy + 1.0, atol=1e-4)
    assert abs(out[0, 0] - a[0, 0]) < 1e-5 and abs(out[-1, -1] - a[-1, -1]) < 1e-5
