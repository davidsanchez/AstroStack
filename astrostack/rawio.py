"""Lecture des images (RAW via rawpy/LibRaw, ou TIFF/PNG/JPEG) et calibration (darks, flats, bias).

Tout est normalisé en RGB linéaire 16 bits (uint16, 0..65535) en sortie de ce module,
avant toute opération d'alignement / empilement.
"""
from __future__ import annotations

import os

import numpy as np

from .timing import Lap, measure

RAW_EXTS = (".cr2", ".cr3", ".dng", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw", ".raw")
STD_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
SUPPORTED_EXTS = RAW_EXTS + STD_EXTS


def list_image_files(folder: str) -> list[str]:
    """Liste triée des fichiers image supportés contenus dans un dossier."""
    out = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
            out.append(os.path.join(folder, name))
    return out


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Décodage sRGB -> linéaire (pour les JPEG/PNG 8 bits)."""
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_raw_linear(path: str, half: bool = False, chrono=None) -> np.ndarray:
    """Charge un fichier RAW et le dématricise en RGB linéaire 16 bits (uint16 HxWx3).

    gamma=(1,1) + no_auto_bright => pas de courbe de gamma : on reste en linéaire,
    ce qui est important pour moyenner correctement les poses.
    half=True : dématriçage à demi-taille (un pixel RGB par quadruplet du Bayer), plus rapide
    mais moins de détail ; réservé au mode « Rapide ».
    """
    import rawpy  # import tardif : pas besoin de rawpy pour les tests unitaires

    lap = Lap(chrono)
    with rawpy.imread(path) as raw:
        lap.mark("ouverture RAW")
        kwargs = dict(
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1.0, 1.0),  # linéaire
            bright=1.0,
            output_color=rawpy.ColorSpace.sRGB,
            half_size=half,
        )
        try:
            rgb = raw.postprocess(dcb_enhance=False, **kwargs)
        except TypeError:  # anciennes versions de rawpy sans dcb_enhance
            rgb = raw.postprocess(**kwargs)
        lap.mark("dématriçage LibRaw")
    out = np.ascontiguousarray(rgb, dtype=np.uint16)
    lap.mark("mise en forme RGB")
    return out


def load_std_linear(path: str) -> np.ndarray:
    """Charge TIFF/PNG/JPEG -> RGB linéaire 16 bits."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile  # TIFF 16 bits RGB : lu directement (PIL ne gère pas l'RGB uint16)

        arr = tifffile.imread(path)
    else:
        import imageio.v3 as iio

        arr = iio.imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] >= 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint16:
        return arr.astype(np.uint16)
    if arr.dtype == np.uint8:
        lin = srgb_to_linear(arr.astype(np.float32) / 255.0)
    else:
        lin = srgb_to_linear(np.clip(arr.astype(np.float32), 0.0, 1.0))
    return (lin * 65535.0).astype(np.uint16)


def load_image_linear(path: str, half: bool = False, chrono=None) -> np.ndarray:
    """Charge n'importe quelle image supportée en RGB linéaire 16 bits.
    half : demi-taille (RAW seulement ; sans effet sur TIFF/PNG/JPEG)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        return load_raw_linear(path, half=half, chrono=chrono)
    with measure(chrono, "lecture TIFF/PNG"):
        return load_std_linear(path)


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTS


# ---------------------------------------------------------------------------
# Calibration sur le Bayer brut (avant dématriçage) — la bonne façon de faire
# la soustraction de dark : on soustrait le master dark sur raw_image_visible,
# puis on dématrice. rawpy permet d'écrire dans raw_image_visible (vue modifiable).
# ---------------------------------------------------------------------------

def load_bayer(path: str):
    """Charge le Bayer brut d'un RAW. Retourne (bayer uint16 HxW, pattern 2x2, black(4,), white)."""
    import rawpy

    with rawpy.imread(path) as raw:
        bayer = raw.raw_image_visible.copy()
        pattern = np.asarray(raw.raw_pattern)
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white = int(raw.white_level)
    return bayer, pattern, black, white


def _black_full(pattern: np.ndarray, black: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Carte du black level (par canal Bayer) à la taille de l'image."""
    m = black[pattern]  # 2x2
    reps = (shape[0] // 2 + 2, shape[1] // 2 + 2)
    return np.tile(m, reps)[: shape[0], : shape[1]].astype(np.float32)


def build_bayer_master(paths: list[str], kind: str, target_shape=None, progress_cb=None):
    """Moyenne de plusieurs Bayers bruts -> (master float32 HxW, pattern, black).

    Toutes les images sont redimensionnées sur target_shape si besoin.
    """
    if not paths:
        return None
    import cv2

    acc = None
    pattern = black = None
    for i, p in enumerate(paths):
        if progress_cb:
            progress_cb(0, i, len(paths), f"Master {kind} (Bayer brut) : {os.path.basename(p)}")
        bayer, pat, blk, _ = load_bayer(p)
        if target_shape is not None and bayer.shape != tuple(target_shape):
            bayer = cv2.resize(bayer, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        acc = bayer.astype(np.float32) if acc is None else acc + bayer
        pattern, black = pat, blk
    return acc / len(paths), pattern, black


def load_raw_calibrated_bayer(path, master_dark=None, master_bias=None, master_flat=None, half=False, chrono=None):
    """Charge un RAW, calibre SUR LE BAYER (dark / bias / flat soustraits avant
    dématriçage), puis dématrice -> RGB linéaire 16 bits (uint16 HxWx3).

    master_* : tuples (master float32 HxW, pattern, black) ou None.
    Retourne None si l'écriture in-place du Bayer échoue (l'appelant retombe
    alors sur la calibration post-dématriçage).

    Gestion du black level : LibRaw soustrait le black level au dématriçage, donc on
    soustrait (dark - black) pour que le net soit exactement light - dark.
    """
    import rawpy

    if master_dark is None and master_bias is None and master_flat is None:
        return load_raw_linear(path, half=half, chrono=chrono)

    lap = Lap(chrono)
    with rawpy.imread(path) as raw:
        vis = raw.raw_image_visible  # la décompression du RAW a lieu ici
        pattern = np.asarray(raw.raw_pattern)
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        h, w = vis.shape
        black_full = _black_full(pattern, black, (h, w))
        lap.mark("lecture RAW (décompression)")

        def _fit(m):
            m = m[0]
            if m.shape != (h, w):
                import cv2

                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
            return m

        x = vis.astype(np.float32)
        if master_bias is not None:
            mb = _fit(master_bias)
            x -= mb - black_full  # bias net du black level
            if master_dark is not None:
                x -= _fit(master_dark) - mb  # dark net du bias
        elif master_dark is not None:
            x -= _fit(master_dark) - black_full  # dark net du black level
        np.clip(x, 0.0, None, out=x)

        if master_flat is not None:
            mf = _fit(master_flat)
            if master_dark is not None:
                base = _fit(master_dark)
            elif master_bias is not None:
                base = _fit(master_bias)
            else:
                base = black_full
            fsig = mf - base
            fsig = fsig / max(float(fsig.mean()), 1e-6)
            x /= np.maximum(fsig, 1e-3)
            np.clip(x, 0.0, 65535.0, out=x)

        try:
            vis[:] = x.astype(vis.dtype)  # écriture in-place dans le buffer LibRaw
        except Exception:
            return None  # vue non modifiable -> fallback post-dématriçage
        lap.mark("calibration Bayer (dark / bias / flat)")

        kwargs = dict(
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1.0, 1.0),
            bright=1.0,
            output_color=rawpy.ColorSpace.sRGB,
            half_size=half,
        )
        try:
            rgb = raw.postprocess(dcb_enhance=False, **kwargs)
        except TypeError:
            rgb = raw.postprocess(**kwargs)
        lap.mark("dématriçage LibRaw")
    out = np.ascontiguousarray(rgb, dtype=np.uint16)
    lap.mark("mise en forme RGB")
    return out


def _resize_like(img: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    import cv2

    if img.shape[:2] != ref_shape:
        img = cv2.resize(img, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_AREA)
    return img


def build_master(paths: list[str], label: str = "", progress_cb=None, target_shape=None) -> np.ndarray | None:
    """Moyenne de plusieurs images (dark / flat / bias) -> master float32 HxWx3.

    Toutes les images sont redimensionnées sur `target_shape` (ou sur la 1ère image si
    non fourni) : aucun crash si des fichiers de tailles différentes sont mélangés.
    """
    if not paths:
        return None
    import cv2

    acc = None
    ref_shape = target_shape
    for i, p in enumerate(paths):
        if progress_cb:
            progress_cb(0, i, len(paths), f"Master {label or 'calibration'} : {os.path.basename(p)}")
        img = load_image_linear(p).astype(np.float32)
        if ref_shape is None:
            ref_shape = img.shape[:2]
        if img.shape[:2] != tuple(ref_shape):
            img = cv2.resize(img, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_AREA)
        acc = img if acc is None else acc + img
    return acc / len(paths)


def calibrate(
    img_u16: np.ndarray,
    master_dark: np.ndarray | None = None,
    master_bias: np.ndarray | None = None,
    master_flat: np.ndarray | None = None,
) -> np.ndarray:
    """Applique bias, dark puis flat sur une image linéaire 16 bits -> float32 0..65535.

    Note : la soustraction du dark se fait ici après dématriçage (v1). Pour une v2,
    soustraire le dark sur le Bayer brut (raw.raw_image_visible) avant dématriçage.
    Les darks doivent être pris à ISO/exposition/température proches des lights.
    """
    x = np.asarray(img_u16, dtype=np.float32)  # copie seulement si le type change
    shape = x.shape[:2]
    if master_bias is not None:
        x -= _resize_like(master_bias, shape)
    if master_dark is not None:
        # si un bias séparé est fourni, le dark est supposé "dark - bias" ; sinon dark brut
        x -= _resize_like(master_dark, shape)
    np.clip(x, 0.0, None, out=x)
    if master_flat is not None:
        f = _resize_like(master_flat, shape)
        f = f / max(float(f.mean()), 1e-6)
        x /= np.maximum(f, 1e-3)
    np.clip(x, 0.0, 65535.0, out=x)  # en place : pas de tableau supplémentaire
    return x
