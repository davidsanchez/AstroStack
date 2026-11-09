"""Test des fantômes d'alignement (v0.9.1) — sans RAW ni Qt :

  python tests/test_ghosting.py

Ciel stellaire + sol sombre et arbre. Chaque pose est recalée sur le ciel, mais le sol
subit une PARALLAXE de quelques px : ses bords se retrouvent en fantôme dans le ciel
voisin lorsque le rejet des pixels aberrants est trop faible. On vérifie que le rejet
robuste (médiane + MAD, mode="robust", opt-in) réduit nettement ce fantôme par rapport à
une simple moyenne. Modèle trépied : sol et arbre FIXES en pixels, seul le ciel dérive.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astrostack.mlmask import refine_foreground
from astrostack.stacker import stack_frames
from astrostack.stars import estimate_alignment, luminance

H, W = 900, 1200
HOR = 640


def make_scene(seed=0):
    rng = np.random.default_rng(seed)
    sky = np.full((H, W, 3), 2200.0, np.float32)
    for _ in range(900):
        x, y = rng.uniform(0, W), rng.uniform(0, HOR)
        amp = rng.uniform(3000, 30000)
        s = rng.uniform(0.8, 1.6)
        x0, y0, r = int(x), int(y), 5
        ys0, ys1 = max(0, y0 - r), min(H, y0 + r + 1)
        xs0, xs1 = max(0, x0 - r), min(W, x0 + r + 1)
        yy, xx = np.mgrid[ys0:ys1, xs0:xs1]
        sky[ys0:ys1, xs0:xs1] += (amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)))[..., None]
    fgm = np.zeros((H, W), np.uint8)
    fgm[HOR:] = 1
    cv2.line(fgm, (600, HOR), (600, 300), 1, thickness=20)
    rb = np.random.default_rng(seed + 5)
    for _ in range(35):
        x0, y0 = 600, int(rb.integers(320, HOR - 20))
        ang = rb.uniform(-np.pi * 0.85, -np.pi * 0.15)
        L = rb.uniform(60, 220)
        cv2.line(fgm, (x0, y0), (int(np.clip(x0 + L * np.cos(ang), 0, W - 1)),
                                 int(np.clip(y0 + L * np.sin(ang), 0, H - 1))), 1, thickness=3)
    fgl = np.full((H, W, 3), 300.0, np.float32)
    return sky, fgm.astype(bool), fgl


def _warp(img, M):
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def make_poses(seed=0, n=5):
    """Trépied : le ciel dérive d'une pose à l'autre, le sol et l'arbre restent fixes en pixels."""
    sky, fgm, fgl = make_scene(seed)
    rng = np.random.default_rng(seed + 11)
    frames = []
    for i in range(n):
        tx, ty = (0.0, 0.0) if i == 0 else tuple(rng.uniform(-6, 6, 2))
        rot = 0.0 if i == 0 else rng.uniform(-0.4, 0.4)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        s_w = _warp(sky, M)
        fr = np.where(fgm[..., None], fgl, s_w) + rng.normal(0, 40, (H, W, 3)).astype(np.float32) + 300
        frames.append(np.clip(fr, 0, 65535).astype(np.float32))
    return frames, sky, fgm, fgl


def ghost_error(out, sky, fgm, fgl):
    truth = np.where(fgm[..., None], fgl, sky) + 300.0
    d = cv2.distanceTransform((~fgm).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (d > 2) & (d <= 14) & (~fgm)
    ring[:HOR - 160] = False
    err = np.abs(out - truth).mean(axis=2)
    return float(err[ring].mean())


def main_check(seed=0):
    frames, sky, fgm, fgl = make_poses(seed)
    u8 = np.clip(frames[0] / 65535.0 * 255.0 * 4, 0, 255).astype(np.uint8)
    prob = cv2.GaussianBlur(cv2.resize(cv2.resize(fgm.astype(np.float32), (W // 16, H // 16),
                                                  interpolation=cv2.INTER_AREA),
                                       (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 1.0)
    fg_w = refine_foreground(prob, u8, lum_refine=True, lum_edge=True, edge_band=40).astype(np.float32) / 255.0
    ref_gray = luminance(frames[0])
    sky_mask = (1.0 - fg_w) > 0.5
    transforms = [None]
    for i in range(1, len(frames)):
        M, _ = estimate_alignment(ref_gray, luminance(frames[i]), sky_mask, refine_ecc=True, refine_local=True)
        transforms.append(M)
    get = lambda i: frames[i]
    mean_out = stack_frames(get, transforms, fg_w, mode="mean")
    robust_out = stack_frames(get, transforms, fg_w, mode="robust", kappa=2.5)
    g_mean = ghost_error(mean_out, sky, fgm, fgl)
    g_rob = ghost_error(robust_out, sky, fgm, fgl)
    print(f"fantôme (moyenne) : {g_mean:.1f}   fantôme (rejet robuste) : {g_rob:.1f}")
    return g_mean, g_rob


def test_robust_rejection_reduces_ghosts():
    g_mean, g_rob = main_check(0)
    assert g_rob < 0.6 * g_mean, f"le rejet robuste doit réduire nettement le fantôme ({g_rob:.1f} vs {g_mean:.1f})"


def main():
    test_robust_rejection_reduces_ghosts()
    print("OK — fantômes d'alignement réduits par le rejet robuste")


if __name__ == "__main__":
    main()
