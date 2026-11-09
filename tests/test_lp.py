"""Tests de la correction de pollution lumineuse (v1.0.0) — sans RAW, sans ONNX, sans Qt :

  python tests/test_lp.py

Le ciel est fourni par un masque de type automatique : une estimation grossière (grille 16 px)
affinée par le même raffinage que le masque ML (bande de luminance + filtre guidé).
Scène : ciel sombre avec gradient coloré (orange sur R), lueur quadratique, étoiles ponctuelles,
avant-plan sombre. La correction doit :
  1. aplatir le gradient dans chaque canal ;
  2. conserver le niveau médian du ciel ;
  3. conserver le contraste des étoiles ;
  4. ne jamais toucher l'avant-plan ;
  5. ne produire ni valeurs négatives, ni force supérieure à 100 %.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astrostack.lpcorrect import correct_light_pollution
from astrostack.mlmask import refine_foreground

H, W = 500, 700
FG_TOP = int(H * 0.75)


def make_scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = 3000.0 * (xx / W) + 2500.0 * (1.0 - yy / H) ** 2
    chans = [4000.0 + 1.4 * grad, 4000.0 + 1.0 * grad, 4000.0 + 0.6 * grad]
    img = np.stack(chans, axis=-1).astype(np.float32) + rng.normal(0, 30, (H, W, 3)).astype(np.float32)
    stars = []
    for _ in range(60):
        x, y = int(rng.integers(16, W - 16)), int(rng.integers(16, FG_TOP - 16))
        stars.append((x, y))
        img[y - 1:y + 2, x - 1:x + 2] += 6000.0
    fgt = np.zeros((H, W), bool)
    fgt[FG_TOP:] = True
    img[FG_TOP:] = 900.0 + rng.normal(0, 40, (H - FG_TOP, W, 3)).astype(np.float32)
    img = np.clip(img, 0, 65535)
    return img, fgt, stars


def auto_like_sky(img: np.ndarray, fgt: np.ndarray) -> np.ndarray:
    """Ciel comme le masque automatique : estimation grossière puis raffinage par luminance."""
    prob = cv2.GaussianBlur(cv2.resize(cv2.resize(fgt.astype(np.float32), (W // 16, H // 16),
                                                  interpolation=cv2.INTER_AREA),
                                       (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 1.0)
    u8 = np.clip(img / 65535.0 * 255.0 * 4, 0, 255).astype(np.uint8)
    fg = refine_foreground(prob, u8, lum_refine=True, lum_edge=True, edge_band=40)
    return 1.0 - fg.astype(np.float32) / 255.0


def grad_of(arr, c):
    top = arr[:FG_TOP - 20, :, c]
    return float(top[:, -40:].mean() - top[:, :40].mean())


def test_gradient_and_sky_level():
    img, fgt, _ = make_scene(0)
    sky = auto_like_sky(img, fgt)
    out = correct_light_pollution(img, sky, strength=1.0)
    for c in range(3):
        g_in, g_out = grad_of(img, c), grad_of(out, c)
        assert abs(g_out) < 0.2 * abs(g_in), f"canal {c} : gradient non aplati ({g_in:.0f} -> {g_out:.0f})"
    med_in = float(np.median(img[:FG_TOP - 20, :, 1]))
    med_out = float(np.median(out[:FG_TOP - 20, :, 1]))
    assert abs(med_out - med_in) < 0.03 * med_in, f"niveau du ciel modifié ({med_in:.0f} -> {med_out:.0f})"
    print(f"gradient G : {grad_of(img, 1):.0f} -> {grad_of(out, 1):.0f} ; ciel {med_in:.0f} -> {med_out:.0f}")


def test_stars_keep_contrast():
    img, fgt, stars = make_scene(1)
    out = correct_light_pollution(img, auto_like_sky(img, fgt), strength=1.0)
    ratios = []
    for x, y in stars[:30]:
        p_in = img[y, x, 1] - np.median(img[y - 12:y + 12, x - 12:x + 12, 1])
        p_out = out[y, x, 1] - np.median(out[y - 12:y + 12, x - 12:x + 12, 1])
        ratios.append(p_out / max(p_in, 1e-6))
    r = float(np.median(ratios))
    print(f"contraste des étoiles conservé : {r:.2f}")
    assert r > 0.9, f"les étoiles ont perdu leur contraste ({r:.2f})"


def test_foreground_untouched_and_no_negatives():
    img, fgt, _ = make_scene(2)
    out = correct_light_pollution(img, auto_like_sky(img, fgt), strength=1.0)
    sigma = max(2.0, min(H, W) / 96.0)
    margin = int(6 * sigma) + 2
    assert np.allclose(out[FG_TOP + margin:], img[FG_TOP + margin:], atol=1e-3), "l'avant-plan a été modifié"
    assert float(out.min()) >= 0.0, "valeurs négatives après correction"


def test_strength_capped_and_background_export():
    img, fgt, _ = make_scene(3)
    sky = auto_like_sky(img, fgt)
    out, bg = correct_light_pollution(img, sky, strength=1.0, return_background=True)
    assert bg is not None and bg.shape == img.shape, "le fond estimé doit avoir la taille de l'image"
    over = correct_light_pollution(img, sky, strength=1.5)
    full = correct_light_pollution(img, sky, strength=1.0)
    assert np.allclose(over, full), "la force doit être bornée à 100 %"


def test_no_sky_no_change():
    img, _, _ = make_scene(4)
    out = correct_light_pollution(img, np.zeros(img.shape[:2], np.float32))
    assert np.array_equal(out, img), "sans ciel détecté, l'image doit rester identique"


def main():
    test_gradient_and_sky_level()
    test_stars_keep_contrast()
    test_foreground_untouched_and_no_negatives()
    test_strength_capped_and_background_export()
    test_no_sky_no_change()
    print("OK — correction de pollution lumineuse validée")


if __name__ == "__main__":
    main()
