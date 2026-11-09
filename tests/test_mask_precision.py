"""Test de précision du masque automatique (v0.9.0) — sans modèle ONNX ni Qt :

  python tests/test_mask_precision.py

Scène synthétique pleine résolution façon photo de nuit : ciel sombre et bruité, plus
lumineux vers l'horizon (lueur), avant-plan (sol + arbre) très sombre avec des branches
fines de 1-2 px. Le modèle ML n'est simulé que par sa sortie GROSSIÈRE (probabilité ciel
sur une grille de 16 px, remontée en cubique), comme SegFormer sur une grande image.
On compare l'ancien raffinage (v0.8.x : seuil Otsu global, médian, morphologie) au nouveau
(seuil local + filtre guidé sur la luminance pleine résolution).
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astrostack.mlmask import guided_refine, refine_foreground

H, W = 600, 800
HORIZON = 400
GRID = 16  # résolution de sortie du modèle, en px de l'image (grille grossière)


def make_scene(seed: int = 0, sky0=0.08, sky1=0.14, fg_lum=0.02, noise=0.02, thick=1):
    """Retourne (img_u8_rgb, truth bool HxW). Avant-plan = sol sombre + arbre à branches fines."""
    rng = np.random.default_rng(seed)
    yy = np.arange(H, dtype=np.float32).reshape(H, 1)
    lum = np.broadcast_to(sky0 + (sky1 - sky0) * (yy / HORIZON), (H, W)).astype(np.float32).copy()
    truth = np.zeros((H, W), np.uint8)
    truth[HORIZON:] = 1  # sol
    cv2.line(truth, (400, HORIZON), (400, 120), 1, thickness=14)  # tronc
    rng_b = np.random.default_rng(seed + 7)
    for _ in range(40):  # branches
        x0, y0 = 400, int(rng_b.integers(130, HORIZON - 10))
        ang = rng_b.uniform(-np.pi * 0.85, -np.pi * 0.15)
        L = rng_b.uniform(40, 150)
        x1 = int(np.clip(x0 + L * np.cos(ang), 0, W - 1))
        y1 = int(np.clip(y0 + L * np.sin(ang), 0, H - 1))
        cv2.line(truth, (x0, y0), (x1, y1), 1, thickness=thick)
    truth = truth.astype(bool)
    lum[truth] = fg_lum
    lum = np.clip(lum + rng.normal(0, noise, (H, W)).astype(np.float32), 0.0, 1.0)
    rgb = np.stack([lum * 255.0, lum * 255.0 * 1.04, lum * 255.0 * 1.10], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8), truth


def coarse_onnx_prob(truth: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Probabilité avant-plan telle que la donnerait un modèle sur grille `grid` (cubique)."""
    small = cv2.resize(truth.astype(np.float32), (W // grid, H // grid), interpolation=cv2.INTER_AREA)
    return np.clip(cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float(inter) / float(max(union, 1))


def branch_recall(pred: np.ndarray, truth: np.ndarray) -> float:
    """Part des pixels de branches (hors tronc) retrouvés."""
    tr = truth.copy()
    tr[120:HORIZON, 393:408] = False  # on ignore le tronc, on mesure les branches
    return float(np.logical_and(pred, tr).sum()) / float(max(tr.sum(), 1))


def old_refine(prob_fg: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Comportement v0.8.x (seuil global, médian + morphologie)."""
    return refine_foreground(prob_fg, img, lum_refine=True, lum_edge=True, edge_band=40,
                             guided=False, local_thr=False) > 127


def new_refine(prob_fg: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Comportement v0.9.0 (défaut : seuil local + filtre guidé)."""
    return refine_foreground(prob_fg, img, lum_refine=True, lum_edge=True, edge_band=40) > 127


def test_guided_does_not_shift_aligned_edge():
    """Sur un bord déjà aligné sur le contour de luminance, le filtre guidé ne doit pas le déplacer."""
    img = np.zeros((120, 160, 3), np.uint8)
    img[:, 80:] = 220
    p = np.zeros((120, 160), np.float32)
    p[:, 80:] = 1.0
    p = cv2.GaussianBlur(p, (0, 0), 1.0)
    q = guided_refine(p, img, radius=3, eps=1e-3)
    assert q.min() >= 0.0 and q.max() <= 1.0
    for row in (30, 60, 90):
        t_p = int(np.where(np.diff((p[row] > 0.5).astype(int)) != 0)[0][0]) + 1
        t_q = int(np.where(np.diff((q[row] > 0.5).astype(int)) != 0)[0][0]) + 1
        assert abs(t_q - t_p) <= 1, f"bord déplacé de {t_p} à {t_q} (ligne {row})"


def test_precision_fine_details_and_dark_scene():
    img, truth = make_scene(3, sky0=0.05, sky1=0.20, fg_lum=0.03, noise=0.025, thick=1)
    prob = coarse_onnx_prob(truth)
    old = old_refine(prob, img)
    new = new_refine(prob, img)
    iou_old, iou_new = iou(old, truth), iou(new, truth)
    rec_old, rec_new = branch_recall(old, truth), branch_recall(new, truth)
    print(f"IoU        : ancien {iou_old:.3f}  ->  nouveau {iou_new:.3f}")
    print(f"branches   : ancien {rec_old:.3f}  ->  nouveau {rec_new:.3f}")
    assert iou_new > iou_old + 0.02, "le raffinage v0.9.0 doit améliorer l'IoU sur fond sombre"
    assert rec_new >= rec_old - 0.01, "les branches fines ne doivent pas être nettement moins bien retrouvées"
    assert iou_new > 0.97, f"IoU trop faible : {iou_new:.3f}"


def first_fg_row(mask: np.ndarray, x: int, lo: int = 330, hi: int = 470):
    idx = np.where(mask[lo:hi, x])[0]
    return lo + int(idx[0]) if len(idx) else None


def test_no_overflow_into_sky():
    """Le masque ne doit pas déborder dans le ciel : sur chaque colonne de l'horizon, son bord
    doit être à ±2 px de la vraie frontière (les débordements étaient de 1-2 px en v0.9.0-dev)."""
    img, truth = make_scene(3, sky0=0.08, sky1=0.14, fg_lum=0.02, noise=0.02, thick=2)
    m = new_refine(coarse_onnx_prob(truth), img)
    bad = []
    for x in range(W):
        a, b = first_fg_row(m, x), first_fg_row(truth, x)
        if a is None or b is None:
            continue
        if abs(a - b) > 2:
            bad.append((x, a, b))
    print(f"colonnes avec bord à plus de 2 px de la vérité : {len(bad)}")
    assert len(bad) <= 5, f"débordement ou retrait du bord : {bad[:10]}"


def main():
    test_guided_does_not_shift_aligned_edge()
    test_precision_fine_details_and_dark_scene()
    test_no_overflow_into_sky()
    print("OK — tests de précision du masque réussis")


if __name__ == "__main__":
    main()
