"""Non-régression PL : coin sans ciel (pas d'éclaircissement) et ciel peint exclu du masque PL.

  python -m pytest -q tests/test_lp_fixes.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astrostack.lpcorrect import combine_painted_sky, estimate_background  # noqa: E402


def test_background_bounded_to_sky_range_in_corner():
    H, W = 900, 600
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lum = 3000 + 2500 * np.exp(-((xx - W) / 200.0) ** 2) + 1800 * (1 - yy / H) ** 3
    img = np.stack([lum, lum * 1.05, lum * 0.9], -1).astype(np.float32)
    sky = np.ones((H, W), np.float32)
    sky[int(H * 0.7):, :] = 0
    bg = estimate_background(img, sky)
    for c in range(3):
        lo = float(img[..., c][sky > 0.99].min())
        corner = bg[int(H * 0.85):, :120, c]
        assert corner.min() >= lo - 1.0, f"canal {c} : fond du coin {corner.min():.0f} sous le ciel minimal {lo:.0f}"


def test_painted_foreground_removed_from_pl_sky():
    sky = np.ones((100, 80), np.float32)
    painted = np.zeros((50, 40), np.uint8)  # même avant-plan peint à demi-résolution
    painted[:, :20] = 255
    s = combine_painted_sky(sky, painted)
    assert s.shape == (100, 80)
    assert s[10, 5] < 0.05, "avant-plan peint doit être exclu du ciel PL"
    assert s[10, 70] > 0.95, "le ciel non peint doit être conservé"
    assert combine_painted_sky(sky, None) is sky or np.array_equal(combine_painted_sky(sky, None), sky)


if __name__ == "__main__":
    test_background_bounded_to_sky_range_in_corner()
    test_painted_foreground_removed_from_pl_sky()
    test_black_border_ignored_by_pl()
    print("OK")


def test_black_border_ignored_by_pl():
    """Une bande noire (bord de recalage) ne doit ni tirer le fond vers le bas, ni être corrigée."""
    from astrostack.lpcorrect import correct_light_pollution
    H, W = 300, 400
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lum = 20000 + 3000 * (1 - yy / H)
    clean_img = np.stack([lum] * 3, -1).astype(np.float32)
    bordered = clean_img.copy()
    bordered[:, :15] = 0.0
    sky = np.ones((H, W), np.float32)
    a = correct_light_pollution(clean_img, sky, strength=1.0, feather=0)
    b = correct_light_pollution(bordered, sky, strength=1.0, feather=0)
    ok = slice(None), slice(40, None)
    assert np.abs(a[ok] - b[ok]).max() < 0.01 * 65535, "la bande noire a changé la correction du ciel"
    assert np.abs(b[:, :15]).max() < 1e-3, "la bande noire ne doit pas être modifiée"
