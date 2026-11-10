"""Détection d'étoiles et estimation des transformations d'alignement (frame -> référence).

Stratégie v0.5.0. Les traînées d'étoiles dans les COINS viennent de la distorsion de
l'objectif : la vraie transformation entre poses est D∘R∘D⁻¹ (D = distorsion), ce qu'aucune
homographie ne peut modéliser — l'erreur résiduelle croît avec la distance au centre.

  1. détection/appage à ÉCHELLE RÉDUITE (DET_MAX px) pour la vitesse ;
  2. translation grossière par corrélation de phase fenêtrée sur la zone de ciel ;
  3. appariement d'étoiles (voisin réciproque + ratio de Lowe) puis HOMOGRAPHIE RANSAC
     (>= 20 pts ; sinon affine >= 8 ; sinon similitude >= 4 ; sinon phase seule) ;
  4. affinage photométrique ECC (cv2.findTransformECC, MOTION_HOMOGRAPHY, masque = ciel) ;
  5. WARP LOCAL par THIN-PLATE SPLINE (TPS) : on garde les appariements dont le résidu
     sous l'homographie est faible, et on ajuste une TPS (src frame -> dst référence) ;
     la carte inverse (référence -> frame) sert de map pour cv2.remap. C'est ce qui corrige
     les coins. Désactivable (refine_local=False / --no-local) ;
  6. si le warp local n'est pas possible (trop peu de points), on rend l'homographie remise
     à pleine résolution : H_full = S⁻¹ · H_s · S avec S = diag(s, s, 1).

Conventions (vérifiées avec la doc OpenCV et tests/test_synthetic.py) :
  - warpAffine/warpPerspective SANS WARP_INVERSE_MAP : dst(x) = src(M⁻¹·x), donc M est
    la carte "forward" (contenu) src -> dst. C'est la convention utilisée ici.
  - AVEC WARP_INVERSE_MAP : dst(x) = src(M·x), M est alors la carte d'échantillonnage
    dst -> src (ne PAS mélanger avec la forward !).
  - phaseCorrelate(a, b) = d  <=>  b(x) = a(x - d) : la carte forward b -> a est la
    translation -d, utilisée SANS WARP_INVERSE_MAP.
  - findTransformECC(template, input, G) retourne G = carte d'échantillonnage
    template -> input (à utiliser avec WARP_INVERSE_MAP) : la forward input -> template
    vaut inv(G).

Retour de estimate_alignment : soit une matrice 3x3 float32 (carte forward frame ->
référence, à utiliser avec cv2.warpPerspective(...) flags PAR DÉFAUT), soit le tuple
("remap", map_x, map_y) (maps float32 pleine résolution à utiliser avec cv2.remap).
warp_to_ref accepte les deux.
"""
from __future__ import annotations

import cv2
import numpy as np

DET_MAX = 1600   # taille max de l'image de détection/appage (réduit la charge CPU/RAM)
MIN_LOCAL = 40   # nombre min d'appariements fiables pour activer le warp local TPS


def luminance(img_f32: np.ndarray) -> np.ndarray:
    g = img_f32.astype(np.float32)
    return 0.2126 * g[..., 0] + 0.7152 * g[..., 1] + 0.0722 * g[..., 2]


def detect_stars(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
    max_stars: int = 1200,
    k_sigma: float = 6.0,
    min_dist: int = 7,
) -> np.ndarray:
    """Détecte les étoiles (pics locaux) -> array (N, 2) de coordonnées (x, y) float32.

    mask : float32 HxW, > 0.5 = zone autorisée (ciel). None = toute l'image.
    """
    g = gray.astype(np.float32)
    m = (mask > 0.5) if mask is not None else np.ones(g.shape, dtype=bool)
    vals = g[m]
    if vals.size == 0:
        return np.zeros((0, 2), np.float32)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    thr = med + k_sigma * max(mad, 1e-3)

    gm = np.where(m, g, -1e9).astype(np.float32)
    k = max(3, int(min_dist) | 1)
    dil = cv2.dilate(gm, np.ones((k, k), np.float32))
    ys, xs = np.nonzero((gm >= thr) & (gm >= dil))
    if ys.size == 0:
        return np.zeros((0, 2), np.float32)
    inten = g[ys, xs]
    order = np.argsort(inten)[::-1][:max_stars]
    return np.stack([xs[order], ys[order]], axis=1).astype(np.float32)


def match_points(
    ref_pts: np.ndarray, pts: np.ndarray, max_dist: float = 40.0, ratio: float = 0.85
) -> tuple[np.ndarray, np.ndarray]:
    """Appariement : plus proche voisin réciproque + test de ratio de Lowe.

    ref_pts : étoiles de la référence ; pts : étoiles de l'image (déjà grossièrement
    recalée). Retourne (src, dst) : src dans l'image, dst dans la référence.
    """
    empty = np.zeros((0, 2), np.float32)
    if len(ref_pts) < 3 or len(pts) < 3:
        return empty, empty.copy()
    d2 = ((pts[:, None, :] - ref_pts[None, :, :]) ** 2).sum(-1)
    nn_pts = d2.argmin(axis=1)
    nn_ref = d2.argmin(axis=0)
    order = np.argsort(d2, axis=1)
    src, dst = [], []
    for i, j in enumerate(nn_pts):
        if nn_ref[j] != i:  # pas réciproque
            continue
        if d2[i, j] > max_dist * max_dist:
            continue
        # test de ratio : le meilleur voisin doit être nettement meilleur que le 2nd
        j2 = order[i, 1] if order[i, 0] == j else order[i, 0]
        if d2[i, j2] > 0 and d2[i, j] > (ratio * ratio) * d2[i, j2]:
            continue
        src.append(pts[i])
        dst.append(ref_pts[j])
    if not src:
        return empty, empty.copy()
    return np.asarray(src, np.float32), np.asarray(dst, np.float32)


def _to3(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        return M
    A = np.eye(3, dtype=np.float64)
    A[:2, :] = M
    return A


def compose(M_outer: np.ndarray, M_inner: np.ndarray) -> np.ndarray:
    """Composition de deux transformations (2x3 ou 3x3) -> 3x3 : M_outer ∘ M_inner."""
    return _to3(M_outer) @ _to3(M_inner)


def phase_translation(ref_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Translation grossière frame -> référence par corrélation de phase fenêtrée.

    Retourne (M, response) avec M = translation 2x3 (carte forward frame -> référence).
    """
    m = mask > 0.5
    bg = float(np.median(ref_gray[m])) if m.any() else 0.0
    a = np.where(m, ref_gray, bg).astype(np.float32)
    b = np.where(m, gray, bg).astype(np.float32)
    h, w = a.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    # phaseCorrelate(a, b) = d  <=>  b(x) = a(x - d) : la carte forward b -> a est la translation -d
    (dx, dy), resp = cv2.phaseCorrelate(a * win, b * win)
    M = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    return M, float(resp)


def _apply3(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    P = (H @ np.hstack([pts, np.ones((len(pts), 1), np.float64)]).T).T
    return P[:, :2] / P[:, 2:3]


def _rms(H: np.ndarray, src: np.ndarray, dst: np.ndarray, inl: np.ndarray) -> float:
    if len(src) == 0 or not inl.any():
        return float("nan")
    res = np.sqrt(((_apply3(H, src) - dst) ** 2).sum(axis=1))
    return float(np.sqrt((res[inl] ** 2).mean()))


def _ecc_refine(
    ref_gray: np.ndarray,
    gray: np.ndarray,
    sky_mask: np.ndarray,
    H_init: np.ndarray,
    max_iters: int = 100,
    eps: float = 1e-5,
) -> tuple[np.ndarray, float]:
    """Affinage photométrique ECC (MOTION_HOMOGRAPHY) sur la zone de ciel.

    H_init : carte forward frame -> référence (3x3). findTransformECC(template=ref,
    input=frame) retourne G = carte template -> input, donc la carte forward cherchée
    vaut inv(G). Retourne (H 3x3 float32, cc) ; en cas d'échec, rend H_init inchangé.
    """
    H0 = _to3(H_init)
    try:
        G = np.linalg.inv(H0)
        if abs(G[2, 2]) > 1e-12:
            G = G / G[2, 2]
        G = np.ascontiguousarray(G, dtype=np.float64)
        mask = (np.asarray(sky_mask) > 0.5).astype(np.uint8) * 255
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
        cc, G = cv2.findTransformECC(
            ref_gray.astype(np.float32),
            gray.astype(np.float32),
            G,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            mask,
            5,
        )
        H = np.linalg.inv(G)
        return H.astype(np.float32), float(cc)
    except cv2.error:
        return np.asarray(H_init, dtype=np.float32), float("nan")


# ---------------------------------------------------------------------------
# Warp local par thin-plate spline (TPS) — corrige la distorsion de l'objectif
# et donc les traînées dans les coins qu'une homographie ne peut pas modéliser.
# ---------------------------------------------------------------------------

def _tps_U(r: np.ndarray) -> np.ndarray:
    """Noyau TPS : U(r) = r^2 * ln(r), U(0) = 0."""
    out = np.zeros_like(r)
    pos = r > 0
    out[pos] = r[pos] ** 2 * np.log(r[pos])
    return out


def _tps_fit(P: np.ndarray, Q: np.ndarray, reg: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Ajuste f : P -> Q (P, Q : (N,2) normalisées ~[0,1]). Retourne (w (N,2), a (3,2))."""
    N = len(P)
    r = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    K = _tps_U(r)
    L = np.zeros((N + 3, N + 3), dtype=np.float64)
    L[:N, :N] = K + reg * np.eye(N)
    # colonnes [1, x, y] (dans cet ordre, comme dans _tps_eval : [1, x, y] @ a)
    L[:N, N] = 1.0
    L[:N, N + 1:N + 3] = P
    L[N, :N] = 1.0
    L[N + 1:N + 3, :N] = P.T
    rhs = np.zeros((N + 3, 2), dtype=np.float64)
    rhs[:N] = Q
    coef = np.linalg.solve(L, rhs)
    return coef[:N], coef[N:]


def _tps_eval(P: np.ndarray, w: np.ndarray, a: np.ndarray, X: np.ndarray, block: int = 4096) -> np.ndarray:
    """Évalue f(X) = U(|X-P|) @ w + [1, x, y] @ a, par blocs pour limiter la RAM."""
    out = np.empty((len(X), 2), dtype=np.float64)
    for i in range(0, len(X), block):
        Xb = X[i:i + block]
        r = np.sqrt(((Xb[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        out[i:i + block] = _tps_U(r) @ w + np.hstack([np.ones((len(Xb), 1)), Xb]) @ a
    return out


def _local_tps_maps(
    det_shape: tuple[int, int],
    src_det: np.ndarray,
    dst_det: np.ndarray,
    s: float,
    full_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Construit les maps cv2.remap pleine résolution à partir des appariements (échelle det).

    src_det : points dans l'image frame (échelle det) ; dst_det : points dans la référence
    (échelle det). On ajuste la TPS inverse g : dst -> src (c'est ce que cv2.remap veut :
    dst(x) = src(map(x))), on l'évalue sur toute la grille de la référence, puis on remet
    les maps à pleine résolution (map_full(x) = map_s(s*x) / s).
    """
    h_s, w_s = det_shape
    w_f, h_f = full_wh
    D = float(max(h_s, w_s))
    P = src_det.astype(np.float64) / D
    Q = dst_det.astype(np.float64) / D
    w_inv, a_inv = _tps_fit(Q, P)  # g : référence -> frame (normalisé)
    yy, xx = np.mgrid[0:h_s, 0:w_s]
    X = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64) / D
    G = _tps_eval(Q, w_inv, a_inv, X)  # coordonnées frame_s normalisées
    mx_s = (G[:, 0] * D).reshape(h_s, w_s).astype(np.float32)
    my_s = (G[:, 1] * D).reshape(h_s, w_s).astype(np.float32)
    mx = cv2.resize(mx_s, (w_f, h_f), interpolation=cv2.INTER_LINEAR) / s
    my = cv2.resize(my_s, (w_f, h_f), interpolation=cv2.INTER_LINEAR) / s
    return mx.astype(np.float32), my.astype(np.float32)


def estimate_alignment(
    ref_gray: np.ndarray,
    gray: np.ndarray,
    sky_mask: np.ndarray,
    max_stars: int = 1200,
    det_max: int = DET_MAX,
    refine_ecc: bool = True,
    refine_local: bool = True,
    min_local: int = MIN_LOCAL,
):
    """Estime la transformation qui envoie l'image courante sur la référence.

    Retourne (M, info) où M est :
      - une matrice 3x3 float32 (carte forward frame -> référence, pleine résolution de
        l'image fournie), à utiliser avec cv2.warpPerspective(...) flags PAR DÉFAUT ;
      - ou le tuple ("remap", map_x, map_y) (maps float32) à utiliser avec cv2.remap,
        quand le warp local TPS est actif (cas par défaut si assez d'étoiles appariées).
    info = {method, inliers, rms, n_ref_stars, n_matched, n_local, phase_resp, ecc_cc,
            det_scale}.
    """
    h, w = ref_gray.shape
    s = min(1.0, det_max / float(max(h, w)))
    if s < 1.0:
        sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        ref_s = cv2.resize(ref_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        gray_s = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(sky_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    else:
        sw, sh = w, h
        ref_s, gray_s, mask_s = ref_gray, gray, sky_mask
    size_s = (sw, sh)

    # 1) translation grossière (gère les grandes dérives) — M_t = forward frame -> référence
    M_t, resp = phase_translation(ref_s, gray_s, mask_s)
    gray_w = cv2.warpAffine(gray_s, M_t, size_s, flags=cv2.INTER_LINEAR)

    # 2) appariement d'étoiles sur l'image grossièrement recalée
    ref_pts = detect_stars(ref_s, mask=mask_s, max_stars=max_stars)
    pts = detect_stars(gray_w, mask=mask_s, max_stars=max_stars)
    src, dst = match_points(ref_pts, pts, max_dist=40.0)

    info: dict = {
        "method": "phase",
        "inliers": 0,
        "rms": float("nan"),
        "n_ref_stars": int(len(ref_pts)),
        "n_matched": int(len(src)),
        "n_local": 0,
        "phase_resp": resp,
        "ecc_cc": float("nan"),
        "det_scale": s,
    }
    H_s = _to3(M_t)  # carte forward frame -> référence, à l'échelle de détection
    method = "phase"
    inl = np.zeros((0,), dtype=bool)
    # coordonnées des appariements dans l'image frame d'origine (annule M_t)
    src_frame = _apply3(np.linalg.inv(H_s), src) if len(src) else src

    # 3) transformation fine par RANSAC, composée avec la translation grossière
    n = len(src)
    if n >= 20:
        Hh, m_in = cv2.findHomography(src, dst, cv2.RANSAC, 2.5, maxIters=10000, confidence=0.999)
        if Hh is not None and m_in is not None and int(m_in.sum()) >= 12:
            H_s = Hh @ H_s
            inl = m_in.ravel().astype(bool)
            method = "homographie"
    if method == "phase" and n >= 8:
        est, m_in = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                                         maxIters=10000, confidence=0.999)
        if est is not None and m_in is not None and int(m_in.sum()) >= 8:
            H_s = _to3(est) @ H_s
            inl = m_in.ravel().astype(bool)
            method = "affine"
    if method == "phase" and n >= 4:
        est, m_in = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                                                maxIters=10000, confidence=0.999)
        if est is not None and m_in is not None and int(m_in.sum()) >= 4:
            H_s = _to3(est) @ H_s
            inl = m_in.ravel().astype(bool)
            method = "similitude"

    if inl.any():
        info["inliers"] = int(inl.sum())
        info["rms"] = _rms(H_s, src_frame, dst, inl)
    info["method"] = method

    # 4) affinage photométrique ECC sur le ciel
    if refine_ecc and method != "phase":
        H_ecc, cc = _ecc_refine(ref_s, gray_s, mask_s, H_s)
        if cc == cc:  # ECC a convergé (cc non NaN)
            H_s = H_ecc
            info["ecc_cc"] = cc
            info["method"] = method + "+ecc"

    # 5) warp local TPS : corrige la distorsion de l'objectif (traînées dans les coins)
    if refine_local and len(src) >= min_local:
        res = np.sqrt(((_apply3(H_s, src_frame) - dst) ** 2).sum(axis=1))
        thr = max(4.0, 3.0 * info["rms"]) if inl.any() else 6.0
        keep = res < thr
        if int(keep.sum()) >= min_local:
            try:
                mx, my = _local_tps_maps((sh, sw), src_frame[keep], dst[keep], s, (w, h))
                info["method"] += "+tps"
                info["n_local"] = int(keep.sum())
                return ("remap", mx, my), info
            except np.linalg.LinAlgError:
                pass  # TPS mal conditionnée -> on garde l'homographie

    # 6) repli : homographie remise à pleine résolution, H_full = S⁻¹ · H_s · S
    S = np.diag([s, s, 1.0])
    H_full = np.linalg.inv(S) @ H_s @ S
    return H_full.astype(np.float32), info


def warp_to_ref(img: np.ndarray, M, size: tuple[int, int]) -> np.ndarray:
    """Recale `img` sur la géométrie de la référence. size = (width, height).

    M : None (identité), matrice 2x3 (affine), 3x3 (homographie), ou tuple
    ("remap", map_x, map_y) produit par le warp local TPS.
    """
    return warp_to_ref_valid(img, M, size)[0]


def warp_to_ref_valid_band(img: np.ndarray, M, size: tuple[int, int], r0: int, r1: int):
    """Comme warp_to_ref_valid, mais ne calcule que les lignes [r0, r1) de la référence.

    Retourne (warp (r1-r0, w, C), validité (r1-r0, w)). Sert à l'empilement par bandes :
    la mémoire de travail reste bornée quelle que soit la taille de l'image.
    """
    w, h = size[0], size[1]
    bh = r1 - r0
    if M is None:
        return img[r0:r1], np.ones((bh, w), np.float32)
    if isinstance(M, tuple) and len(M) == 3 and M[0] == "remap":
        mx, my = np.asarray(M[1], dtype=np.float32), np.asarray(M[2], dtype=np.float32)
        if mx.shape != (h, w):
            mx = cv2.resize(mx, (w, h), interpolation=cv2.INTER_LINEAR)
            my = cv2.resize(my, (w, h), interpolation=cv2.INTER_LINEAR)
        mx, my = np.ascontiguousarray(mx[r0:r1]), np.ascontiguousarray(my[r0:r1])
        warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # la source est l'image ENTIÈRE (les coordonnées de source décident de la validité)
        valid = cv2.remap(
            np.ones((h, w), np.float32), mx, my, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return warped, (valid > 0.5).astype(np.float32)
    ones = np.ones((h, w), np.float32)
    Mf = np.asarray(M, dtype=np.float64)
    if Mf.shape == (3, 3):
        # décalage de la sortie : la bande [r0, r1) devient les lignes 0..bh-1
        T = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -float(r0)], [0.0, 0.0, 1.0]])
        M2 = T @ Mf
        warp = lambda src: cv2.warpPerspective(
            src, M2, (w, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    else:
        M2 = Mf.copy()
        M2[1, 2] -= r0
        warp = lambda src: cv2.warpAffine(
            src, M2, (w, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    return warp(img), (warp(ones) > 0.5).astype(np.float32)


def warp_to_ref_valid(img: np.ndarray, M, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Comme warp_to_ref, mais retourne aussi le masque de validité (float32 HxW,
    1.0 = le pixel recalé tombe DANS l'image source, 0.0 = zone de bord remplie à 0).

    Indispensable pour l'empilement : un 0 de bordure ne doit pas être moyenné comme
    un vrai pixel noir (sinon bande sombre sur les bords de l'image empilée).
    """
    h, w = size[1], size[0]
    if M is None:
        return img, np.ones((h, w), np.float32)
    if isinstance(M, tuple) and len(M) == 3 and M[0] == "remap":
        mx, my = np.asarray(M[1], dtype=np.float32), np.asarray(M[2], dtype=np.float32)
        if mx.shape != (h, w):
            mx = cv2.resize(mx, (w, h), interpolation=cv2.INTER_LINEAR)
            my = cv2.resize(my, (w, h), interpolation=cv2.INTER_LINEAR)
        warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.remap(
            np.ones((h, w), np.float32), mx, my, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        return warped, (valid > 0.5).astype(np.float32)
    ones = np.ones((h, w), np.float32)
    if np.asarray(M).shape == (3, 3):
        warped = cv2.warpPerspective(
            img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        valid = cv2.warpPerspective(
            ones, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    else:
        warped = cv2.warpAffine(
            img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        valid = cv2.warpAffine(
            ones, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    return warped, (valid > 0.5).astype(np.float32)
