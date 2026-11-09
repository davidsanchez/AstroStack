"""Tests de la correction de pollution lumineuse (v0.9.3) — sans RAW ni Qt :

  python tests/test_lp.py

Scène : ciel sombre avec gradient coloré (orange sur R, bleuté sur B), lueur quadratique
vers l'horizon, étoiles ponctuelles, avant-plan sombre. On vérifie que la correction :
  1. aplatit le gradient dans chaque canal ;
  2. conserve le niveau médian du ciel ;
  3. ne détruit pas les étoiles (pics conservés) ;
  4. ne touche pas l'avant-plan ;
  5. ne produit ni valeurs négatives ni écrasement des ombres.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astrostack.lpcorrect import correct_light_pollution

H, W = 500, 700
FG_TOP = int(H * 0.75)


def make_scene(seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 4000.0
    grad = 3000.0 * (xx / W) + 2500.0 * (1.0 - yy / H) ** 2       # gradient + lueur quadratique
    chans = [
        base + 1.4 * grad,   # R : pollution orange
        base + 1.0 * grad,   # G
        base + 0.6 * grad,   # B : pollution moins bleue
    ]
    img = np.stack(chans, axis=-1).astype(np.float32)
    img += rng.normal(0, 30, img.shape).astype(np.float32)
    stars = []
    for _ in range(60):  # étoiles ponctuelles
        x, y = int(rng.integers(16, W - 16)), int(rng.integers(16, FG_TOP - 16))
        stars.append((x, y))
        img[y - 1:y + 2, x - 1:x + 2] += 6000.0
    fg = np.full((H - FG_TOP, W, 3), 900.0, np.float32) + rng.normal(0, 40, (H - FG_TOP, W, 3)).astype(np.float32)
    img[FG_TOP:] = fg
    img = np.clip(img, 0, 65535)
    sky = np.zeros((H, W), np.float32)
    sky[:FG_TOP] = 1.0
    return img, sky, stars


def grad_of(arr, c):
    top = arr[:FG_TOP - 20, :, c]
    return float(top[:, -40:].mean() - top[:, :40].mean())


def test_lp_gradient_and_levels():
    img, sky, stars = make_scene(0)
    out = correct_light_pollution(img, method="synthetic", strength=1.0, sky=sky, sky_only=True)
    for c in range(3):
        g_in, g_out = grad_of(img, c), grad_of(out, c)
        assert abs(g_out) < 0.2 * abs(g_in), f"canal {c} : gradient non aplati ({g_in:.0f} -> {g_out:.0f})"
    med_in = float(np.median(img[:FG_TOP - 20, :, 1]))
    med_out = float(np.median(out[:FG_TOP - 20, :, 1]))
    assert abs(med_out - med_in) < 0.03 * med_in, f"niveau du ciel modifié ({med_in:.0f} -> {med_out:.0f})"
    print(f"gradients canal G : {grad_of(img, 1):.0f} -> {grad_of(out, 1):.0f} ; "
          f"médiane ciel {med_in:.0f} -> {med_out:.0f}")


def test_stars_preserved():
    img, sky, stars = make_scene(1)
    out = correct_light_pollution(img, method="synthetic", strength=1.0, sky=sky, sky_only=True)
    ratios = []
    for x, y in stars[:30]:
        peak_in = img[y, x, 1] - np.median(img[y - 12:y + 12, x - 12:x + 12, 1])
        peak_out = out[y, x, 1] - np.median(out[y - 12:y + 12, x - 12:x + 12, 1])
        ratios.append(peak_out / max(peak_in, 1e-6))
    r = float(np.median(ratios))
    print(f"contraste des étoiles conservé : {r:.2f}")
    assert r > 0.9, f"les étoiles ont perdu leur contraste ({r:.2f})"


def test_foreground_untouched_and_no_negatives():
    img, sky, _ = make_scene(2)
    out = correct_light_pollution(img, method="synthetic", strength=1.0, sky=sky, sky_only=True)
    sigma = max(2.0, min(H, W) / 96.0)  # même marge que test_synthetic : le fondu s'étend en 6 sigma
    margin = int(6 * sigma) + 2
    assert np.allclose(out[FG_TOP + margin:], img[FG_TOP + margin:], atol=1e-3), "l'avant-plan a été modifié"
    assert float(out.min()) >= 0.0, "valeurs négatives après correction"


def test_background_export_and_strength_cap():
    img, sky, _ = make_scene(3)
    out, bg = correct_light_pollution(img, method="synthetic", strength=1.0, sky=sky, sky_only=True,
                                      return_background=True)
    assert bg.shape == img.shape, "le fond estimé doit avoir la taille de l'image"
    out_over = correct_light_pollution(img, method="synthetic", strength=1.5, sky=sky, sky_only=True)
    out_full = correct_light_pollution(img, method="synthetic", strength=1.0, sky=sky, sky_only=True)
    assert np.allclose(out_over, out_full), "la force doit être bornée à 100 %"


def main():
    test_lp_gradient_and_levels()
    test_stars_preserved()
    test_foreground_untouched_and_no_negatives()
    test_background_export_and_strength_cap()
    print("OK — correction de pollution lumineuse validée")


if __name__ == "__main__":
    main()
