"""Pipeline complet : calibration (darks/flats/bias) -> détection d'étoiles -> alignement
sur la référence -> empilement ciel recalé + avant-plan fixe -> export TIFF 16 bits.

Fonction unique `run_pipeline`, utilisée par l'interface graphique (Qt) et par la CLI.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .imaging import tonemap_u16
from .rawio import (
    DEMOSAIC_CHOICES,
    build_bayer_master,
    build_master,
    calibrate,
    is_raw,
    load_bayer,
    load_image_linear,
    load_raw_calibrated_bayer,
)
from .stacker import feather, stack_frames
from .timing import Chrono, Phases
from .stars import estimate_alignment, luminance


def _save_result(out_path: str, result_f32: np.ndarray, fg_w: np.ndarray, fg_base: np.ndarray | None = None) -> dict:
    import imageio.v3 as iio
    import tifffile

    u16 = np.clip(result_f32, 0.0, 65535.0).round().astype(np.uint16)
    tifffile.imwrite(out_path, u16, photometric="rgb")
    base, _ = os.path.splitext(out_path)
    preview_path = base + "_preview.png"
    iio.imwrite(preview_path, tonemap_u16(result_f32))
    mask_path = base + "_mask.png"
    if fg_base is not None:
        # base (masque ONNX / peint) en BLANC, extension par dilatation en ORANGE, ciel en noir
        base_b = np.asarray(fg_base) > 127
        full_b = np.asarray(fg_w) > 0.5
        rgb = np.zeros(full_b.shape + (3,), np.uint8)
        rgb[base_b & full_b] = (255, 255, 255)
        rgb[full_b & ~base_b] = (255, 140, 0)
        iio.imwrite(mask_path, rgb)
    else:
        iio.imwrite(mask_path, (fg_w * 255.0).round().astype(np.uint8))
    return {"tif": out_path, "preview": preview_path, "mask": mask_path}


class CancelledError(RuntimeError):
    """Levée quand l'utilisateur demande l'annulation (paramètre should_cancel)."""


def run_pipeline(
    lights: list[str],
    darks: list[str] | None = None,
    flats: list[str] | None = None,
    biases: list[str] | None = None,
    mask: np.ndarray | None = None,
    scale: float = 1.0,
    mode: str = "sigma",
    kappa: float = 2.5,
    refine_ecc: bool = True,
    refine_local: bool = True,
    stack_grow: int = 5,
    ref_index: int = 0,
    out_path: str | None = None,
    lp_correct: bool = False,
    lp_strength: float = 1.0,
    lp_feather: float | None = None,
    lp_top_gradient: bool = True,
    lp_top_boost: float = 0.5,
    lp_target: str = "median",
    lp_shared_shape: bool = True,
    lp_lum_refine: bool = True,
    lp_lum_thr: float | None = None,
    lp_edge_band: int = 40,
    lp_fit_exclude=None,
    meta_out: dict | None = None,
    progress_cb=None,
    should_cancel=None,
    fast: bool = False,
    load_threads: int = 2,
    demosaic: str = "auto",
) -> np.ndarray:
    """Exécute tout le pipeline. Retourne l'image résultat float32 HxWx3 (0..65535).

    lights      : liste de fichiers (RAW CR2/DNG/... ou TIFF/PNG) ; lights[ref_index] = référence
                  (défaut 0 = la 1ère ; l'UI met l'index du milieu).
    darks/flats/biases : listes de fichiers de calibration (optionnelles).
    darks/flats/biases : listes de fichiers de calibration (optionnelles).
    mask        : uint8 HxW (255 = avant-plan), à n'importe quelle taille (redimensionné),
                  ou None = tout le ciel est recalé (pas d'avant-plan).
    scale       : 1.0 = pleine résolution, 0.5 = moitié, etc. (réduit RAM/CPU).
    mode        : "sigma" (sigma clipping) ou "mean" (moyenne simple).
    refine_ecc  : affinage photométrique ECC de l'alignement sur le ciel (corrige les
                  traînées résiduelles, notamment sur les bords). Désactivable (--no-ecc).
    refine_local: warp local par thin-plate spline (TPS) à partir des étoiles appariées,
                  réparties sur tout le champ (coins compris) — corrige la distorsion de
                  l'objectif, cause classique des traînées dans les coins. Désactivable
                  (--no-local).
    out_path    : si fourni, écrit le TIFF 16 bits + un aperçu PNG + le masque PNG.
    lp_correct  : correction de la pollution lumineuse (v0.8.0), EN LINÉAIRE, après
                  l'empilement et avant l'export : soustraction d'un fond estimé.
    lp_strength : force 0..1 (défaut 1.0 = correction complète).
    lp_feather  : sigma (px) du lissage de la transition au bord du masque ciel — v0.8.2,
                  None = auto (min(h,l)/96, anti-halos), 0 = masque dur.
    meta_out    : dict optionnel rempli avec les données du stack (v0.8.4) : "stack_raw"
                  (résultat AVANT correction PL), "sky"/"fg_w" (poids à la résolution de
                  travail), "work_shape" — permet de réappliquer la correction PL sans
                  refaire l'empilement.
    progress_cb : callback(stage:int, i:int, n:int, msg:str).
    should_cancel : callback() -> bool ; si True, lève CancelledError (annulation douce).
    fast        : mode « Rapide » (False = « Précis », défaut). Peut modifier légèrement le
                  résultat : RAW décodés à demi-taille si scale <= 50 %, pas de recalcul des
                  poses en sigma clipping, arithmétique allégée.
    load_threads: nombre de RAW lus/dématricés en parallèle (lights et masters). Ne change
                  pas le résultat ; plus de mémoire (une image calibrée en plus par thread).
    demosaic    : "auto" (AHD en Précis, PPG en Rapide), "ahd", "ppg" ou "linear" (RAW seulement).
    """
    if demosaic not in DEMOSAIC_CHOICES:
        raise ValueError(f"Dématriçage inconnu : {demosaic}")
    if load_threads < 1:
        raise ValueError("load_threads doit être >= 1")
    dm = ("ppg" if fast else None) if demosaic == "auto" else demosaic

    def cb(stage, i, n, msg=""):
        if should_cancel is not None and should_cancel():
            raise CancelledError("Annulation demandée par l'utilisateur.")
        if progress_cb:
            progress_cb(stage, i, n, msg)

    if not lights:
        raise ValueError("Aucune image light fournie.")
    chrono = Chrono()  # temps de calibration par étape (affiché en fin d'empilement)
    phases = Phases()  # temps réel par phase du run (affiché à la fin)
    align_chrono = Chrono()  # sous-étapes de l'alignement (cumulées sur les images)
    if not (0 <= ref_index < len(lights)):
        ref_index = 0
    ref_path = lights[ref_index]

    # --- image de référence (lights[ref_index]) ---
    cb(0, 0, len(lights), f"Chargement de la référence ({os.path.basename(ref_path)})…")
    # --- masters de calibration ---
    # Si la référence et tous les fichiers de calibration sont des RAW :
    # on calibre SUR LE BAYER BRUT (dark/bias/flat avant dématriçage) — plus correct.
    # Sinon (TIFF/PNG, ou calibration non RAW) : calibration post-dématriçage en RGB.
    calib_files = (darks or []) + (flats or []) + (biases or [])
    use_bayer = is_raw(ref_path) and all(is_raw(p) for p in calib_files)
    master_dark = master_bias = master_flat = None
    bayer_dark = bayer_bias = bayer_flat = None
    # mode rapide : RAW à demi-taille quand la résolution de travail est <= 50 %
    # (seul le décodage RAW sait produire la demi-taille ; TIFF/PNG restent pleine taille)
    half = bool(fast) and scale <= 0.5 and is_raw(ref_path)
    ref_pre = None  # référence décodée UNE seule fois (calibration post-dématriçage)
    ref_shape = None
    if not use_bayer:
        try:
            ref_pre = load_image_linear(ref_path, half=half, chrono=chrono, demosaic=dm)
        except Exception as exc:
            raise RuntimeError(f"Impossible de lire la référence {ref_path} : {exc}") from exc
        ref_shape = ref_pre.shape[:2]
    if use_bayer:
        target = load_bayer(ref_path)[0].shape  # taille du Bayer de la référence
        with chrono.measure("masters Bayer (dark / bias / flat)"):
            if darks:
                bayer_dark = build_bayer_master(darks, "dark", target, cb, workers=load_threads)
            if biases:
                bayer_bias = build_bayer_master(biases, "bias", target, cb, workers=load_threads)
            if flats:
                bayer_flat = build_bayer_master(flats, "flat", target, cb, workers=load_threads)
        cb(0, 0, 1, "Calibration sur Bayer brut (dark/bias/flat soustraits avant dématriçage)")
    else:
        with chrono.measure("masters RGB (dark / bias / flat)"):
            master_dark = build_master(darks or [], "dark", cb, target_shape=ref_shape, workers=load_threads)
            master_bias = build_master(biases or [], "bias", cb, target_shape=ref_shape, workers=load_threads)
            master_flat = build_master(flats or [], "flat", cb, target_shape=ref_shape, workers=load_threads)

    # masters post-dématriçage, construits à la demande uniquement si le repli en a besoin
    # (chemin Bayer dont l'écriture in-place a échoué) : sans eux, l'image n'était pas calibrée
    fallback_lock = threading.Lock()
    fallback_masters: list = []

    def _fallback_masters():
        with fallback_lock:
            if not fallback_masters:
                with chrono.measure("masters RGB (repli)"):
                    fallback_masters.append((
                        build_master(darks or [], "dark"),
                        build_master(biases or [], "bias"),
                        build_master(flats or [], "flat"),
                    ))
            return fallback_masters[0]

    def load_light(path: str) -> np.ndarray:
        """Charge une light calibrée (Bayer si possible, sinon RGB post-dématriçage)."""
        if use_bayer:
            img = load_raw_calibrated_bayer(path, bayer_dark, bayer_bias, bayer_flat, half=half, chrono=chrono,
                                            demosaic=dm)
            if img is not None:
                return img
            # repli : calibration post-dématriçage avec les masters RGB
            md, mb, mf = _fallback_masters()
            rgb = load_image_linear(path, half=half, chrono=chrono, demosaic=dm)
            with chrono.measure("calibration (repli, après dématriçage)"):
                return calibrate(rgb, md, mb, mf)
        rgb = load_image_linear(path, half=half, chrono=chrono, demosaic=dm)
        with chrono.measure("calibration (après dématriçage)"):
            return calibrate(rgb, master_dark, master_bias, master_flat)

    if ref_pre is not None:
        ref = np.asarray(calibrate(ref_pre, master_dark, master_bias, master_flat), dtype=np.float32)
        del ref_pre
    else:
        ref = np.asarray(load_light(ref_path), dtype=np.float32)
    h, w = ref.shape[:2]
    if half:  # décodé à demi-taille : dimensions pleine résolution = double
        h, w = 2 * h, 2 * w
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    if ref.shape[:2] != (nh, nw):
        ref = cv2.resize(ref, (nw, nh), interpolation=cv2.INTER_AREA)
    wh = ref.shape[:2]

    # --- masque d'avant-plan (peint par l'utilisateur), adouci ---
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 3:  # masque RGB -> on prend un canal
            mask = mask[..., 0]
        m = cv2.resize(mask, (wh[1], wh[0]), interpolation=cv2.INTER_NEAREST)
        # les petits îlots (étoiles détectées comme avant-plan) restent fixes à l'empilement
        from .mlmask import drop_small_islands
        m = drop_small_islands(m)
        m_raw_painted = m.copy()  # masque peint sans dilatation ni fondu, pour la PL
        # marge de sécurité : on dilate l'avant-plan de ~0,5 % (min 3 px) pour que le bord
        # du masque — et son fondu — tombent DANS LE CIEL, jamais sur l'avant-plan
        # (sinon les pixels d'avant-plan au bord du masque seraient warpés = fantômes)
        k = max(3, int(round(min(wh) * 0.005))) | 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
        fg_w = feather(m, sigma=max(2.0, min(wh) * 0.004))
        # marge de stacking (2 px par défaut) — elle ne sert QU'À l'empilement :
        # ni dans le masque affiché/exporté (fg_w), ni dans la correction de pollution lumineuse.
        if stack_grow > 0:
            m_stack = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=int(stack_grow))
            fg_stack = feather(m_stack, sigma=max(2.0, min(wh) * 0.004))
        else:
            fg_stack = fg_w
    else:
        fg_w = np.zeros(wh, dtype=np.float32)
        fg_stack = fg_w
    sky = 1.0 - fg_w

    n = len(lights)
    cache: dict[int, np.ndarray] = {}
    CACHE_MAX_BYTES = 2_500_000_000  # ~2.5 Go de frames calibrées en RAM
    cache_bytes = 0
    # v1.2.1 : la référence est déjà calibrée et à la bonne taille : pas de second décodage
    cache[ref_index] = ref
    cache_bytes = ref.nbytes
    # v1.2.1 : lecture de l'image suivante en arrière-plan (lecture disque / décodage RAW)
    loader = ThreadPoolExecutor(max_workers=load_threads)
    pending: dict[int, object] = {}

    def _load_work(i: int) -> np.ndarray:
        try:
            img_lin = load_light(lights[i])
            with chrono.measure("conversion float32"):
                img = np.asarray(img_lin, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Impossible de lire {lights[i]} : {exc}") from exc
        if img.shape[:2] != wh:  # tailles différentes (autre appareil, JPEG, etc.) -> on redimensionne
            img = cv2.resize(img, (wh[1], wh[0]), interpolation=cv2.INTER_AREA)
        return img

    def prefetch(i: int) -> None:
        """Lance la lecture des images i, i+1, … (jusqu'à load_threads en attente)."""
        for j in range(i, i + load_threads):
            if (0 <= j < n and j not in cache and j not in pending and len(pending) < load_threads
                    and cache_bytes < CACHE_MAX_BYTES):
                pending[j] = loader.submit(_load_work, j)

    def _get_frame(i: int) -> np.ndarray:
        nonlocal cache_bytes
        if i in cache:
            return cache[i]
        cb(1, i, n, f"Calibration {os.path.basename(lights[i])}")
        fut = pending.pop(i, None)
        img = fut.result() if fut is not None else _load_work(i)
        if cache_bytes < CACHE_MAX_BYTES:
            cache[i] = img
            cache_bytes += img.nbytes
        return img

    def get_frame(i: int) -> np.ndarray:
        prefetch(i + 1)  # l'image suivante se lit pendant le calcul de celle-ci
        return _get_frame(i)

    # --- alignement de chaque light sur la référence (phase + étoiles + homographie + ECC + TPS) ---
    phases.mark("préparation")
    ref_gray = luminance(ref)
    transforms: list = [None] * n  # la référence est déjà dans sa géométrie
    for i in range(n):
        if i == ref_index:
            continue
        if should_cancel is not None and should_cancel():
            raise CancelledError("Annulation demandée par l'utilisateur.")
        img = get_frame(i)
        g = luminance(img)
        M, info = estimate_alignment(ref_gray, g, sky, refine_ecc=refine_ecc, refine_local=refine_local,
                                     chrono=align_chrono)
        transforms[i] = M
        rms = f"{info['rms']:.2f} px" if info["inliers"] else "n/a"
        cc = f", cc ECC {info['ecc_cc']:.3f}" if info.get("ecc_cc") == info.get("ecc_cc") else ""
        loc = f", {info['n_local']} pts TPS" if info.get("n_local") else ""
        cb(
            1,
            i,
            n,
            f"Alignement {i + 1}/{n} : {os.path.basename(lights[i])} — "
            f"méthode {info['method']}, {info['inliers']} inliers, rms {rms}{cc}{loc}, "
            f"{info['n_ref_stars']} étoiles détectées",
        )

    phases.mark("alignement")
    cb(2, n, n, f"Temps d'alignement (cumulés sur les images) — {align_chrono.summary()}")
    # --- empilement ---
    try:
        result = stack_frames(get_frame, transforms, fg_stack, mode=mode, kappa=kappa, progress_cb=cb,
                              fast=bool(fast))
    finally:
        loader.shutdown(wait=False, cancel_futures=True)
    cb(4, n, n, f"Temps de calibration (cumulés sur les images) — {chrono.summary()}")
    phases.mark("empilement")
    if meta_out is not None:  # données pour réappliquer la PL sans re-stacker
        meta_out["stack_raw"] = result
        meta_out["sky"] = sky
        meta_out["fg_w"] = fg_w
        meta_out["work_shape"] = wh

    # --- correction de la pollution lumineuse (linéaire, post-empilement) ---
    # le ciel vient du MASQUE AUTOMATIQUE calculé sur le stack (géométrie de référence).
    lp_bg = None
    if lp_correct:
        if should_cancel is not None and should_cancel():
            raise CancelledError("Annulation demandée par l'utilisateur.")
        from .lpcorrect import correct_light_pollution, pl_sky_mask

        cb(3, 0, 1, f"Correction de la pollution lumineuse — force {lp_strength:.0%}, ciel détecté par le masque automatique…")
        try:
            # le masque PL reprend le réglage « Gradient Ciel favorisé » de l'interface
            # le masque PL est calculé sur la RÉFÉRENCE (comme le masque affiché), pas sur le stack :
            # l'affinage luminance (seuil Otsu) donnait sinon des îlots sombres dans le ciel du stack.
            sky_lp = pl_sky_mask(ref, wh, m_raw_painted if mask is not None else None,
                                 top_gradient=lp_top_gradient, top_boost=lp_top_boost,
                                 lum_refine=lp_lum_refine, lum_thr=lp_lum_thr, edge_band=lp_edge_band,
                                 debug_cb=lambda m: cb(3, 0, 1, m))
        except ImportError as exc:
            raise RuntimeError(f"Détection du ciel : {exc}") from exc
        if meta_out is not None:
            meta_out["sky_lp"] = sky_lp
            meta_out["ref_path"] = lights[ref_index]
            meta_out["fg_raw"] = m_raw_painted if mask is not None else None  # pour « Réappliquer la PL »
        if out_path:
            base, ext = os.path.splitext(out_path)
            try:  # diagnostic : masque ciel utilisé
                import imageio.v3 as iio

                iio.imwrite(base + "_maskPL.png", (np.clip(sky_lp, 0.0, 1.0) * 255.0).round().astype(np.uint8))
                cb(3, 0, 1, f"Masque PL écrit : {base}_maskPL.png — ciel = {float((sky_lp > 0.5).mean()):.0%} de l'image")
            except Exception:
                pass
            pre_path = base + "_avant_PL" + (ext or ".tif")  # image AVANT correction (une seule fois)
            if not os.path.exists(pre_path):
                _save_result(pre_path, result, fg_w, fg_base=(m_raw_painted if mask is not None else None))
                cb(3, 0, 1, f"Image AVANT correction PL écrite : {pre_path}")
            else:
                cb(3, 0, 1, f"Image AVANT correction PL déjà existante, conservée : {pre_path}")
        result, lp_bg = correct_light_pollution(
            result, sky_lp, strength=lp_strength, feather=lp_feather,
            return_background=True, debug_cb=lambda m: cb(3, 0, 1, m),
            shared_shape=lp_shared_shape, target=lp_target,
            fit_exclude=(None if lp_fit_exclude is None else cv2.resize(np.asarray(lp_fit_exclude), (wh[1], wh[0]), interpolation=cv2.INTER_NEAREST) > 127),
        )
        cb(3, 1, 1, "Correction de la pollution lumineuse terminée.")

    phases.mark("pollution lumineuse")
    if out_path:
        paths = _save_result(out_path, result, fg_w, fg_base=(m_raw_painted if mask is not None else None))
        cb(4, n, n, f"Écrit : {paths['tif']}")
        if lp_correct and lp_bg is not None:  # fond de pollution estimé, étiré pour l'œil
            import imageio.v3 as iio

            base0, _ = os.path.splitext(out_path)
            b = np.mean(lp_bg, axis=2) if lp_bg.ndim == 3 else lp_bg
            lo, hi = np.percentile(b, [1, 99.5])
            disp = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
            iio.imwrite(base0 + "_fondPL.png", (disp * 255).round().astype(np.uint8))
            cb(4, n, n, f"Fond de pollution lumineuse écrit : {base0}_fondPL.png")
    phases.mark("écriture")
    cb(4, n, n, f"Temps réel par phase — {phases.summary()}")
    return result
