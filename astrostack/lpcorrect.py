"""Correction de la pollution lumineuse — v1.0.0 (réécriture).

Appliquée EN LINÉAIRE, après l'empilement, avant l'export 16 bits.

1. CIEL : détecté par le MASQUE AUTOMATIQUE (ONNX affiné, mlmask.foreground_mask_from_linear),
   calculé sur l'image empilée (déjà dans la géométrie de référence, donc sans alignement ni
   référence à relire). Le masque peint n'intervient pas ici.
2. FOND : estimé sur les pixels de ciel par un ajustement polynomial robuste. La FORME du gradient
   est commune aux trois couleurs ; chaque canal n'a que son amplitude et son décalage. Les étoiles
   et valeurs aberrantes sont rejetées par itérations.
3. SOUSTRACTION : la pollution est additive. La cible est le niveau MÉDIAN du ciel : le niveau du
   ciel ne change pas, seul son gradient disparaît. Un plancher doux évite d'écraser les ombres.
4. TRANSITION : la correction n'agit que sur le ciel, avec une transition douce au bord du masque.
   Une zone avant-plan n'est jamais modifiée.
"""
from __future__ import annotations

import cv2
import numpy as np

LP_MAX_STRENGTH = 1.0  # la force est bornée à 100 %


def auto_sky_weight(img_linear: np.ndarray, out_hw: tuple[int, int] | None = None, debug_cb=None,
                    top_gradient: bool = True, top_boost: float = 0.5, lum_refine: bool = True,
                    lum_thr: float | None = None, edge_band: int = 40) -> np.ndarray:
    """Poids de ciel float32 HxW (1 = ciel, 0 = avant-plan) : EXACTEMENT le masque ML de l'interface
    (dégradé « haut = ciel », affinage luminance, seuil, bord recalé, filtre guidé, sans marge).

    v1.1.5 : la version « ONNX brut » gardait les nuages lumineux classés avant-plan (îlots) que
    l'affinage luminance de l'interface corrige. La PL reprend donc les mêmes réglages que le masque
    affiché. Lève ImportError si onnxruntime n'est pas installé.
    """
    from .mlmask import foreground_mask_from_linear

    fg = foreground_mask_from_linear(
        img_linear.astype(np.float32),
        max_side=2000,  # même résolution de travail que le masque de l'interface (défaut 2000 px)
        debug_cb=debug_cb,
        grow=0,
        lum_refine=lum_refine,
        lum_thr=lum_thr,
        lum_edge=edge_band > 0,
        edge_band=edge_band,
        top_gradient=top_gradient,
        top_boost=top_boost,
    )
    if out_hw is not None and fg.shape[:2] != tuple(out_hw):
        fg = cv2.resize(fg, (int(out_hw[1]), int(out_hw[0])), interpolation=cv2.INTER_LINEAR)
    return (1.0 - fg.astype(np.float32) / 255.0).astype(np.float32)


def exclude_dark_border(sky: np.ndarray, img: np.ndarray, level: float, margin: int = 3) -> np.ndarray:
    """Retire du ciel PL les zones noires (bords de recalage, sans donnée) et leur voisinage.

    Une bande quasi nulle sur les trois canaux n'est pas du ciel : elle tirerait le fond polynomial
    vers le bas (et éclaircirait les coins). On la retire, plus une marge de `margin` px.
    """
    im = img if img.ndim == 3 else img[..., None]
    black = (im.max(axis=2) <= 0.001 * level).astype(np.uint8)
    if not black.any():
        return sky
    black = cv2.dilate(black, np.ones((3, 3), np.uint8), iterations=int(margin))
    return (np.asarray(sky, dtype=np.float32) * (1.0 - black.astype(np.float32))).astype(np.float32)


def combine_painted_sky(sky: np.ndarray, painted_fg: np.ndarray | None) -> np.ndarray:
    """Ciel final pour la PL : le masque automatique, AMPUTÉ de l'avant-plan peint par l'utilisateur.

    painted_fg : masque d'avant-plan peint (fg_w, 1 = avant-plan), même si sa taille diffère (redimensionné).
    Ce qui est peint en avant-plan n'est jamais considéré comme ciel, même si le modèle ML le croit.
    Le masque peint utilisé est celui affiché (sans la marge d'empilement de 2 px).
    """
    sky = np.asarray(sky, dtype=np.float32)
    if painted_fg is None:
        return sky
    p = np.asarray(painted_fg, dtype=np.float32)
    if p.ndim == 3:
        p = p[..., 0]
    if p.max() > 1.5:
        p = p / 255.0
    if p.shape[:2] != sky.shape[:2]:
        p = cv2.resize(p, (sky.shape[1], sky.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(sky * (1.0 - np.clip(p, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)


def _poly_basis(h: int, w: int, deg: int) -> np.ndarray:
    """Base polynomiale 2D (coordonnées normalisées -1..1), forme (h*w, K)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xn = xx / max(w - 1, 1) * 2.0 - 1.0
    yn = yy / max(h - 1, 1) * 2.0 - 1.0
    cols = [(xn ** i) * (yn ** j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    return np.stack(cols, axis=-1).reshape(h * w, -1)


def _robust_lstsq(X: np.ndarray, y: np.ndarray, iters: int = 4, k: float = 3.0) -> np.ndarray:
    """Moindres carrés avec rejet itératif des résidus aberrants (étoiles, bruit fort)."""
    keep = np.ones(len(y), bool)
    coef = None
    for _ in range(iters):
        if keep.sum() < 3 * X.shape[1]:
            break
        coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        res = y - X @ coef
        scale = 1.4826 * float(np.median(np.abs(res[keep]))) + 1e-9
        keep = np.abs(res) <= k * scale
    if coef is None:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def estimate_background(img: np.ndarray, sky_w: np.ndarray, deg: int = 3, max_small: int = 256,
                        shared_shape: bool = False, exclude: np.ndarray | None = None) -> np.ndarray | None:
    """Fond de pollution HxWxC estimé sur le ciel, ou None si le ciel est insuffisant.

    v1.1.0 (défaut, shared_shape=False) : CORRECTION PAR CANAL — chaque canal (R, G, B) reçoit son
    propre polynôme, ajusté sur ses propres pixels de ciel. Chaque gradient de couleur est ainsi
    capturé, même lorsque sa forme diffère d'une couleur à l'autre.
    shared_shape=True : ancien comportement (forme commune + amplitude et décalage par canal).
    """
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = img[..., None]
    s = min(1.0, max_small / max(h, w))
    sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = small[..., None]
    sky_small = cv2.resize(sky_w.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) >= 0.99
    if exclude is not None:  # v1.1.5 : zone exclue du calcul (nuages) : jamais un échantillon du fond
        ex_small = cv2.resize(np.asarray(exclude, dtype=np.float32), (sw, sh), interpolation=cv2.INTER_AREA) >= 0.5
        sky_small = sky_small & ~ex_small
    # v1.1.4 : enveloppe basse par bloc. Un nuage est plus clair que le ciel voisin : dans chaque
    # bloc on ne garde que les pixels les plus sombres (30e percentile de luminance). Le gradient de
    # pollution, lent, est conservé ; la lueur des nuages n'attire plus le fond.
    lum_s = small.mean(axis=-1) if small.ndim == 3 else small[..., 0]
    blk_px = 16
    keep = np.zeros_like(sky_small)
    for y0 in range(0, sh, blk_px):
        for x0 in range(0, sw, blk_px):
            blk = sky_small[y0:y0 + blk_px, x0:x0 + blk_px]
            if int(blk.sum()) < 8:
                continue
            lv = lum_s[y0:y0 + blk_px, x0:x0 + blk_px]
            thr = float(np.percentile(lv[blk], 30))
            keep[y0:y0 + blk_px, x0:x0 + blk_px] = blk & (lv <= thr)
    idx = keep.ravel()
    if int(idx.sum()) < 50:
        return None
    B = _poly_basis(sh, sw, deg)
    bg_small = np.empty(small.shape, dtype=np.float32)
    if shared_shape:
        lum = small.mean(axis=-1).ravel()
        cs = _robust_lstsq(B[idx], lum[idx])
        S = (B @ cs).reshape(sh, sw)
        Ssk = S.ravel()[idx]
        X2 = np.stack([Ssk, np.ones_like(Ssk)], axis=1)
        for c in range(small.shape[2]):
            ab = _robust_lstsq(X2, small[..., c].ravel()[idx])
            bg_small[..., c] = ab[0] * S + ab[1]
    else:
        for c in range(small.shape[2]):
            cs = _robust_lstsq(B[idx], small[..., c].ravel()[idx])
            bg_small[..., c] = (B @ cs).reshape(sh, sw)
    # Le polynôme s'extrapole mal là où il n'y a pas de ciel (coins, sol) : il peut descendre ou
    # monter bien au-delà des niveaux réellement observés. On borne le fond à la plage des
    # échantillons de ciel, canal par canal, pour qu'aucun coin ne s'éclaircisse artificiellement.
    for c in range(small.shape[2]):
        vals = small[..., c].ravel()[idx]
        lo, hi = float(vals.min()), float(vals.max())
        bg_small[..., c] = np.clip(bg_small[..., c], lo, hi)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    if bg.ndim == 2:
        bg = bg[..., None]
    return bg.astype(np.float32)


def soft_floor(x: np.ndarray, eps: float) -> np.ndarray:
    """Plancher doux : identique au-dessus de eps ; en dessous, décroissance exponentielle vers 0
    (monotone et continue). Évite l'écrêtage brutal à zéro qui écrase les ombres."""
    eps = max(float(eps), 1e-9)
    low = eps * np.exp((np.minimum(x, eps) - eps) / eps)  # borné : pas de dépassement
    return np.where(x >= eps, x, low).astype(np.float32)


def sky_transition(sky: np.ndarray, feather: float | None = None) -> np.ndarray:
    """Poids de correction float32 HxW, transition douce au bord du masque.

    - masque déjà adouci (ML affiné) : utilisé tel quel, sa transition est déjà là ;
    - masque dur : léger flou court, SANS dilatation (v1.1.4) : l'ancienne dilatation de ~60 px
      faisait déborder la correction dans le sol et les arbres (bande claire au bord du sol) ;
    - feather=0 : masque dur, sans transition.
    """
    wgt = np.clip(np.asarray(sky, dtype=np.float32), 0.0, 1.0)
    soft_frac = float(((wgt > 0.05) & (wgt < 0.95)).mean())
    if feather is None and soft_frac > 0.02:
        return wgt
    sigma = max(1.5, min(wgt.shape[:2]) / 800.0) if feather is None else float(feather)
    if sigma > 0:
        wgt = cv2.GaussianBlur(wgt, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(wgt, 0.0, 1.0)


def correct_light_pollution(
    img: np.ndarray,
    sky: np.ndarray,
    strength: float = 1.0,
    feather: float | None = None,
    deg: int = 3,
    neutral: bool = True,
    shared_shape: bool = True,  # v1.1.3 : forme commune par défaut (évite la dominante bleue sous les nuages chauds)
    fit_exclude: np.ndarray | None = None,  # v1.1.5 : zone exclue du calcul du fond (reste corrigée)
    target: str = "median",  # v1.1.3 : "median" (défaut, niveau médian conservé) ou "dark" (ciel le plus sombre conservé)
    return_background: bool = False,
    debug_cb=None,
):
    """Corrige le gradient de pollution lumineuse de l'image linéaire `img` (HxWxC).

    sky      : poids de ciel HxW (1 = ciel) — issu du masque automatique (auto_sky_weight).
    strength : 0..1 (borné à LP_MAX_STRENGTH).
    feather  : sigma de la transition (None = auto, 0 = masque dur).
    Retourne l'image corrigée, ou (image, fond estimé) si return_background.
    """
    img = img.astype(np.float32)
    level = 65535.0 if float(img.max()) > 1.5 else 1.0
    strength = float(np.clip(strength, 0.0, LP_MAX_STRENGTH))
    sky = exclude_dark_border(sky, img, level)  # bords noirs : jamais pris pour du ciel
    sky_full = sky >= 0.5
    if not sky_full.any():
        if debug_cb:
            debug_cb("PL : aucun ciel détecté, correction ignorée.")
        return (img, None) if return_background else img

    bg = estimate_background(img, sky, deg=deg, shared_shape=shared_shape, exclude=fit_exclude)
    if bg is None:
        if debug_cb:
            debug_cb("PL : ciel insuffisant pour estimer le fond, correction ignorée.")
        return (img, None) if return_background else img

    # Cible par canal. Le niveau médian du ciel est conservé. Avec neutral=True (défaut), les trois
    # médianes sont ramenées à une même valeur : la dominante de couleur du ciel disparaît (fond neutre).
    med_img = np.array([float(np.median(img[..., c][sky_full])) for c in range(img.shape[2])])
    med_bg = np.array([float(np.median(bg[..., c][sky_full])) for c in range(img.shape[2])])
    if target == "dark":
        # v1.1.3 : cible = bas de la distribution du fond (2e percentile) : seule la pollution est
        # retirée, le ciel le plus sombre reste tel quel. Les zones très éclairées sont plus assombries.
        dark_ref = np.array([float(np.percentile(bg[..., c][sky_full], 2)) for c in range(img.shape[2])])
        targets = np.full_like(dark_ref, dark_ref.mean()) if neutral else dark_ref
    elif neutral:
        level_ref = float(med_img.mean())
        targets = level_ref - med_img + med_bg
    else:
        targets = med_bg.copy()
    out = img.copy()
    for c in range(img.shape[2]):
        out[..., c] = img[..., c] - strength * (bg[..., c] - targets[c])
    out = soft_floor(out, eps=0.002 * level)
    wgt = sky_transition(sky, feather)[..., None]
    out = out * wgt + img * (1.0 - wgt)
    if debug_cb:
        debug_cb(
            "PL : fond polynomial par canal (degré {}), {} ({}), force {:.0%}".format(
                deg, "fond neutre" if neutral else "niveau par canal",
                ", ".join(f"{t:.0f}" for t in targets), strength
            )
            + f" (niveaux 0..{level:.0f})"
        )
    return (out, bg) if return_background else out
