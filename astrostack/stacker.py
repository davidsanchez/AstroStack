"""Empilement des images recalées, avec séparation ciel / avant-plan.

- La zone de ciel (1 - masque) est empilée à partir des images RECALÉES sur la référence
  (les étoiles s'alignent, le bruit diminue en racine de N).
- L'avant-plan (masque peint par l'utilisateur) est empilé SANS recalage : il reste fixe
  et net, comme dans Starry Landscape Stacker.
- Le raccord est adouci par un masque "feathered" (flou gaussien sur le bord du masque).

v0.6.0 : chaque frame recalée est pondérée par son MASQUE DE VALIDITÉ (les pixels qui
tombent hors de l'image source après warp ne comptent pas) — sinon les zéros de bordure
étaient moyennés comme de vrais pixels noirs et produisaient une bande sombre sur les bords
de l'image empilée. Là où aucune frame ne couvre (cnt = 0), on retombe sur la référence.

Deux passes : passe 1 = moyenne/écart-type par pixel dans le ciel (poids de validité) ;
passe 2 = sigma clipping (rejet des pixels déviants : satellites, avions, lucioles...)
puis moyenne.
"""
from __future__ import annotations

import numpy as np

from .stars import warp_to_ref_valid


def feather(mask_u8: np.ndarray, sigma: float) -> np.ndarray:
    """Masque uint8 (255 = avant-plan) -> float32 0..1 adouci (bords en dégradé)."""
    import cv2

    m = mask_u8.astype(np.float32) / 255.0
    k = int(max(3, round(sigma))) | 1
    return cv2.GaussianBlur(m, (k, k), sigma)


def stack_frames(
    get_frame,
    transforms: list,
    fg_w: np.ndarray,
    mode: str = "sigma",
    kappa: float = 2.5,
    progress_cb=None,
) -> np.ndarray:
    """Empile les images.

    get_frame(i) -> image float32 HxWx3 calibrée (i dans [0, n[).
    transforms[i] -> matrice 2x3/3x3 forward frame->référence, tuple ("remap", mx, my),
                     ou None (référence elle-même).
    fg_w : float32 HxW, 1.0 = avant-plan (fixe), 0.0 = ciel (recalé).
    Retourne l'image float32 HxWx3 (0..65535).
    """
    h, w = fg_w.shape
    size = (w, h)
    sky3 = (1.0 - fg_w)[..., None]
    fg3 = fg_w[..., None]
    n = len(transforms)

    def cb(stage, i, msg):
        if progress_cb:
            progress_cb(stage, i, n, msg)

    # --- passe 1 : statistiques dans le ciel (avec poids de validité) + avant-plan ---
    acc = np.zeros((h, w, 3), np.float32)
    acc2 = np.zeros((h, w, 3), np.float32)
    cnt = np.zeros((h, w, 1), np.float32)
    fg_acc = np.zeros((h, w, 3), np.float32)
    fg_cnt = np.zeros((h, w, 1), np.float32)
    ref_img = None

    for i in range(n):
        img = get_frame(i).astype(np.float32)
        if img.shape[:2] != (h, w):
            raise ValueError(
                f"Image {i} : taille {img.shape[:2]} différente de la référence {(h, w)} "
                "(le redimensionnement automatique a échoué)."
            )
        wimg, val = warp_to_ref_valid(img, transforms[i], size)  # val : 1.0 = pixel couvert
        v3 = val[..., None]
        if i == 0:
            ref_img = img  # la référence couvre tout le canevas, jamais warpée
        acc += wimg * sky3 * v3
        acc2 += (wimg * wimg) * sky3 * v3
        cnt += sky3 * v3
        fg_acc += img * fg3
        fg_cnt += fg3
        cb(2, i, f"Statistiques ciel {i + 1}/{n}")

    safe = cnt > 0  # (h, w, 1)
    cnt_safe = np.maximum(cnt, 1e-6)
    mean = np.where(safe, acc / cnt_safe, 0.0).astype(np.float32)
    var = np.where(safe, np.maximum(acc2 / cnt_safe - mean ** 2, 0.0), 0.0).astype(np.float32)
    std = np.sqrt(var)
    del acc2

    # --- passe 2 : sigma clipping dans le ciel ---
    if mode == "sigma":
        acc3 = np.zeros_like(acc)
        cnt3 = np.zeros((h, w, 1), np.float32)
        for i in range(n):
            img = get_frame(i).astype(np.float32)
            if img.shape[:2] != (h, w):
                raise ValueError(f"Image {i} : taille {img.shape[:2]} != référence {(h, w)}")
            wimg, val = warp_to_ref_valid(img, transforms[i], size)
            valid = (np.abs(wimg - mean) <= kappa * std).all(axis=2, keepdims=True) & safe & (val[..., None] > 0.5)
            acc3 += wimg * valid * sky3
            cnt3 += valid * sky3
            cb(3, i, f"Sigma clipping {i + 1}/{n}")
        # là où le clipping a tout rejeté : moyenne de la passe 1 ; où aucune frame ne
        # couvre : la référence (non warpée) — jamais de bande noire
        fallback = np.where(safe, mean, ref_img if ref_img is not None else 0.0)
        sky_img = np.where(cnt3 > 0, acc3 / np.maximum(cnt3, 1e-6), fallback)
    else:
        sky_img = np.where(safe, mean, ref_img if ref_img is not None else 0.0)

    fg_img = np.where(fg_cnt > 0, fg_acc / np.maximum(fg_cnt, 1e-6), 0.0)

    # --- composition finale : ciel recalé + avant-plan fixe, raccord fondu ---
    return sky_img * sky3 + fg_img * fg3
