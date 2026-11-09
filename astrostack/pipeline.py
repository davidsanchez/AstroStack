"""Pipeline complet : calibration (darks/flats/bias) -> détection d'étoiles -> alignement
sur la référence -> empilement ciel recalé + avant-plan fixe -> export TIFF 16 bits.

Fonction unique `run_pipeline`, utilisée par l'interface graphique (Qt) et par la CLI.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .imaging import tonemap_u16
from .rawio import (
    build_bayer_master,
    build_master,
    calibrate,
    is_raw,
    load_bayer,
    load_image_linear,
    load_raw_calibrated_bayer,
)
from .stacker import feather, stack_frames
from .stars import estimate_alignment, luminance


def _save_result(out_path: str, result_f32: np.ndarray, fg_w: np.ndarray) -> dict:
    import imageio.v3 as iio
    import tifffile

    u16 = np.clip(result_f32, 0.0, 65535.0).round().astype(np.uint16)
    tifffile.imwrite(out_path, u16, photometric="rgb")
    base, _ = os.path.splitext(out_path)
    preview_path = base + "_preview.png"
    iio.imwrite(preview_path, tonemap_u16(result_f32))
    mask_path = base + "_mask.png"
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
    stack_grow: int = 2,
    ref_index: int = 0,
    out_path: str | None = None,
    lp_correct: bool = False,
    lp_method: str = "synthetic",
    lp_strength: float = 1.0,
    lp_sky_only: bool = True,
    lp_feather: float | None = None,
    lp_mask_source: str = "painted",
    meta_out: dict | None = None,
    progress_cb=None,
    should_cancel=None,
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
    lp_method   : "synthetic" (défaut : fond lissé « synthetic flat » estimé sur le ciel)
                  ou "graxpert" (modèle ONNX GraXpert, nécessite onnxruntime).
    lp_strength : force 0..1 (défaut 1.0 = correction complète).
    lp_sky_only : si True, la correction n'est appliquée que sur le ciel (masque avant-plan).
    lp_feather  : sigma (px) du lissage de la transition au bord du masque ciel — v0.8.2,
                  None = auto (min(h,l)/96, anti-halos), 0 = masque dur.
    lp_mask_source : masque utilisé POUR LA CORRECTION PL (v0.8.5) : "painted" (masque peint
                  de l'UI) ou "ml" (masque ONNX depuis la référence). v0.8.7 : en mode "ml",
                  le masque ONNX est utilisé BRUT — sans dilatation (+2 px) ni dégradé
                  vertical, réservés au masque d'empilement.
    meta_out    : dict optionnel rempli avec les données du stack (v0.8.4) : "stack_raw"
                  (résultat AVANT correction PL), "sky"/"fg_w" (poids à la résolution de
                  travail), "work_shape" — permet de réappliquer la correction PL sans
                  refaire l'empilement.
    progress_cb : callback(stage:int, i:int, n:int, msg:str).
    should_cancel : callback() -> bool ; si True, lève CancelledError (annulation douce).
    """

    def cb(stage, i, n, msg=""):
        if should_cancel is not None and should_cancel():
            raise CancelledError("Annulation demandée par l'utilisateur.")
        if progress_cb:
            progress_cb(stage, i, n, msg)

    if not lights:
        raise ValueError("Aucune image light fournie.")
    if not (0 <= ref_index < len(lights)):
        ref_index = 0
    ref_path = lights[ref_index]

    # --- image de référence (lights[ref_index]) ---
    cb(0, 0, len(lights), f"Chargement de la référence ({os.path.basename(ref_path)})…")
    try:
        ref_shape = load_image_linear(ref_path).shape[:2]
    except Exception as exc:
        raise RuntimeError(f"Impossible de lire la référence {ref_path} : {exc}") from exc

    # --- masters de calibration ---
    # Si la référence et tous les fichiers de calibration sont des RAW :
    # on calibre SUR LE BAYER BRUT (dark/bias/flat avant dématriçage) — plus correct.
    # Sinon (TIFF/PNG, ou calibration non RAW) : calibration post-dématriçage en RGB.
    calib_files = (darks or []) + (flats or []) + (biases or [])
    use_bayer = is_raw(ref_path) and all(is_raw(p) for p in calib_files)
    master_dark = master_bias = master_flat = None
    bayer_dark = bayer_bias = bayer_flat = None
    if use_bayer:
        target = load_bayer(ref_path)[0].shape  # taille du Bayer de la référence
        if darks:
            bayer_dark = build_bayer_master(darks, "dark", target, cb)
        if biases:
            bayer_bias = build_bayer_master(biases, "bias", target, cb)
        if flats:
            bayer_flat = build_bayer_master(flats, "flat", target, cb)
        cb(0, 0, 1, "Calibration sur Bayer brut (dark/bias/flat soustraits avant dématriçage)")
    else:
        master_dark = build_master(darks or [], "dark", cb, target_shape=ref_shape)
        master_bias = build_master(biases or [], "bias", cb, target_shape=ref_shape)
        master_flat = build_master(flats or [], "flat", cb, target_shape=ref_shape)

    def load_light(path: str) -> np.ndarray:
        """Charge une light calibrée (Bayer si possible, sinon RGB post-dématriçage)."""
        if use_bayer:
            img = load_raw_calibrated_bayer(path, bayer_dark, bayer_bias, bayer_flat)
            if img is not None:
                return img
            # repli : calibration post-dématriçage
        return calibrate(load_image_linear(path), master_dark, master_bias, master_flat)

    ref = load_light(ref_path).astype(np.float32)
    h, w = ref.shape[:2]
    if scale != 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        ref = cv2.resize(ref, (nw, nh), interpolation=cv2.INTER_AREA)
    wh = ref.shape[:2]

    # --- masque d'avant-plan (peint par l'utilisateur), adouci ---
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 3:  # masque RGB -> on prend un canal
            mask = mask[..., 0]
        m = cv2.resize(mask, (wh[1], wh[0]), interpolation=cv2.INTER_NEAREST)
        # marge de sécurité : on dilate l'avant-plan de ~0,5 % (min 3 px) pour que le bord
        # du masque — et son fondu — tombent DANS LE CIEL, jamais sur l'avant-plan
        # (sinon les pixels d'avant-plan au bord du masque seraient warpés = fantômes)
        k = max(3, int(round(min(wh) * 0.005))) | 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
        fg_w = feather(m, sigma=max(2.0, min(wh) * 0.004))
        # v0.9.2 : marge de stacking (2 px par défaut) — elle ne sert QU'À l'empilement :
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

    def get_frame(i: int) -> np.ndarray:
        nonlocal cache_bytes
        if i in cache:
            return cache[i]
        cb(1, i, n, f"Calibration {os.path.basename(lights[i])}")
        try:
            img = load_light(lights[i]).astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"Impossible de lire {lights[i]} : {exc}") from exc
        if img.shape[:2] != wh:  # tailles différentes (autre appareil, JPEG, etc.) -> on redimensionne
            img = cv2.resize(img, (wh[1], wh[0]), interpolation=cv2.INTER_AREA)
        if cache_bytes < CACHE_MAX_BYTES:
            cache[i] = img
            cache_bytes += img.nbytes
        return img

    # --- alignement de chaque light sur la référence (phase + étoiles + homographie + ECC + TPS) ---
    ref_gray = luminance(ref)
    transforms: list = [None] * n  # la référence est déjà dans sa géométrie
    for i in range(n):
        if i == ref_index:
            continue
        if should_cancel is not None and should_cancel():
            raise CancelledError("Annulation demandée par l'utilisateur.")
        img = get_frame(i)
        g = luminance(img)
        M, info = estimate_alignment(ref_gray, g, sky, refine_ecc=refine_ecc, refine_local=refine_local)
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

    # --- empilement ---
    result = stack_frames(get_frame, transforms, fg_stack, mode=mode, kappa=kappa, progress_cb=cb)
    if meta_out is not None:  # v0.8.4 : données pour réappliquer la PL sans re-stacker
        meta_out["stack_raw"] = result
        meta_out["sky"] = sky
        meta_out["fg_w"] = fg_w
        meta_out["work_shape"] = wh

    # --- correction de la pollution lumineuse (linéaire, post-empilement) ---
    lp_bg = None
    if lp_correct:
        if should_cancel is not None and should_cancel():
            raise CancelledError("Annulation demandée par l'utilisateur.")
        from .lpcorrect import correct_light_pollution

        cb(3, 0, 1, f"Correction de la pollution lumineuse — méthode « {lp_method} », force {lp_strength:.0%}…")
        if lp_method == "graxpert" and lp_sky_only:
            lp_sky_only = False  # v0.8.10 : GraXpert corrige TOUTE l'image (le modèle protège l'avant-plan, y compris la lueur d'horizon)
            cb(3, 0, 1, "Méthode GraXpert : correction appliquée à TOUTE l'image (« ciel uniquement » ignoré).")
        sky_lp = sky
        if lp_mask_source == "ml":  # v0.8.5 : masque PL = masque ONNX (depuis la référence)
            from .mlmask import foreground_mask_from_linear

            fg_ml = foreground_mask_from_linear(
                ref, debug_cb=lambda m: cb(3, 0, 1, m),
                top_gradient=False, top_boost=0.5, grow=0,  # v0.8.7 : pas de dilatation ni de dégradé pour la PL
                guided=False, smooth=1, proba_blur=0, morph=False,          # v0.8.9 + v0.9.0 : pas de filtre guidé (PL inchangée) : masque ONNX 100 % BRUT (aucun lissage ni morphologie)
            )
            m_ml = cv2.resize(fg_ml, (wh[1], wh[0]), interpolation=cv2.INTER_NEAREST)
            sky_lp = 1.0 - feather(m_ml, sigma=max(2.0, min(wh) * 0.004))
            cb(3, 0, 1, "Masque PL : masque ML ONNX brut (seuil 0,5, sans lissage ni dilatation).")
        if meta_out is not None:
            meta_out["sky_lp"] = sky_lp
            meta_out["ref_path"] = lights[ref_index]
        if out_path:  # v0.8.9 : sauvegarde du masque PL réellement utilisé (diagnostic : la lueur d'horizon y est-elle classée ciel ?)
            try:
                import imageio.v3 as iio

                base0, _ = os.path.splitext(out_path)
                iio.imwrite(base0 + "_maskPL.png", (np.clip(sky_lp, 0.0, 1.0) * 255.0).round().astype(np.uint8))
                cb(3, 0, 1, f"Masque PL écrit : {base0}_maskPL.png — ciel = {float((sky_lp > 0.5).mean()):.0%} de l'image")
            except Exception:
                pass
        if out_path:  # v0.8.4 : image AVANT correction PL ; v0.8.8 : une seule fois (si déjà là, on la conserve)
            base, ext = os.path.splitext(out_path)
            pre_path = base + "_avant_PL" + (ext or ".tif")
            if not os.path.exists(pre_path):
                _save_result(pre_path, result, fg_w)
                cb(3, 0, 1, f"Image AVANT correction PL écrite : {pre_path}")
            else:
                cb(3, 0, 1, f"Image AVANT correction PL déjà existante, conservée : {pre_path}")
        try:
            lp_ret = correct_light_pollution(
                result,
                method=lp_method,
                strength=lp_strength,
                sky=sky_lp,
                sky_only=lp_sky_only,
                feather=lp_feather,
                debug_cb=lambda m: cb(3, 0, 1, m),
                return_background=bool(out_path),  # v0.9.3 : fond estimé exporté pour contrôle visuel
            )
            if isinstance(lp_ret, tuple):
                result, lp_bg = lp_ret
            else:
                result, lp_bg = lp_ret, None
        except ImportError as exc:
            raise RuntimeError(
                "La méthode ML GraXpert a besoin de onnxruntime : pip install onnxruntime"
            ) from exc
        cb(3, 1, 1, "Correction de la pollution lumineuse terminée.")

    if out_path:
        paths = _save_result(out_path, result, fg_w)
        cb(4, n, n, f"Écrit : {paths['tif']}")
        if lp_correct and lp_bg is not None:  # v0.9.3 : fond de pollution estimé, étiré pour l'œil
            import imageio.v3 as iio

            base0, _ = os.path.splitext(out_path)
            b = np.mean(lp_bg, axis=2) if lp_bg.ndim == 3 else lp_bg
            lo, hi = np.percentile(b, [1, 99.5])
            disp = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
            iio.imwrite(base0 + "_fondPL.png", (disp * 255).round().astype(np.uint8))
            cb(4, n, n, f"Fond de pollution lumineuse écrit : {base0}_fondPL.png")
    return result
