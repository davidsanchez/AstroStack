"""Correction de la pollution lumineuse (dégradé de fond de ciel) — v0.8.0.

Deux méthodes, appliquées EN LINÉAIRE, après l'empilement, avant l'export 16 bits :

- "synthetic" (DÉFAUT) : fond synthétique lisse (« synthetic flat » — sous-échantillonnage
  + flou gaussien large), estimé de préférence sur le ciel (masque), puis SOUSTRACTION
  (la pollution lumineuse est additive) avec une cible = percentile du fond sur le ciel,
  par canal. Rapide, déterministe, zéro modèle.

- "graxpert" : modèle ONNX officiel GraXpert « background extraction » v1.0.1 (~218 Mo,
  licence CC BY-NC-SA 4.0 — usage NON COMMERCIAL), exécuté avec onnxruntime (déjà requis
  pour le masque ML). Prétraitement identique à GraXpert (resize 240x240 + pad 8 -> 256,
  normalisation médiane/MAD * 0.04, clip [-1, 1], entrée NHWC "gen_input_image"), puis
  soustraction du fond estimé. Téléchargé au premier usage et mis en cache dans
  ~/.astrostack/.

Variables d'environnement :
  ASTROSTACK_GRAXPERT_MODEL : chemin local du modèle .onnx
  ASTROSTACK_GRAXPERT_URL   : URL de téléchargement du modèle
"""
from __future__ import annotations

import os

import cv2
import numpy as np

DEFAULT_GRAXPERT_URL = (
    "https://huggingface.co/jromanghf/graxpert-models/resolve/main/graxpert-background-1.0.1.onnx"
)
DEFAULT_GRAXPERT_PATH = os.path.join(os.path.expanduser("~"), ".astrostack", "graxpert-background-1.0.1.onnx")

_graxpert_session = None
_graxpert_path: str | None = None


def graxpert_model_path() -> str:
    return os.environ.get("ASTROSTACK_GRAXPERT_MODEL") or DEFAULT_GRAXPERT_PATH


def ensure_graxpert_model(path: str | None = None, url: str | None = None, progress_cb=None) -> str:
    """Retourne le chemin du modèle GraXpert, en le téléchargeant si absent."""
    path = path or graxpert_model_path()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = url or os.environ.get("ASTROSTACK_GRAXPERT_URL") or DEFAULT_GRAXPERT_URL
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if progress_cb:
        progress_cb(
            "Téléchargement du modèle GraXpert background-extraction 1.0.1 (~218 Mo) — "
            "licence CC BY-NC-SA 4.0 (usage non commercial)…"
        )
    import urllib.request

    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def _synthetic_background(img: np.ndarray, sky: np.ndarray | None = None, max_small: int = 768) -> np.ndarray:
    """Fond lisse estimé par sous-échantillonnage + flou gaussien (synthetic flat).

    v0.8.3 : si `sky` est fourni, le fond est estimé UNIQUEMENT sur les pixels du ciel
    (flou normalisé masqué : blur(img*sky) / blur(sky)) — sinon l'avant-plan lumineux
    (ex. balle de foin) contamine le fond et la soustraction crée un halo autour de
    l'avant-plan. Là où il n'y a que de l'avant-plan (den <= 0.05), on retombe sur le
    flou global.

    v0.8.5 : le masque ciel est d'abord ÉRODÉ (sigma/2) avant le flou normalisé, so on
    n'utilise que du ciel « pur » pour estimer le fond — un pixel d'avant-plan lumineux
    proche du bord ne contamine plus jamais le fond estimé (halo brillant collé au bord
    de l'avant-plan). Si le ciel est trop fin (érodé < 20 % du ciel), on garde tout le ciel.
    """
    h, w = img.shape[:2]
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    s = min(1.0, max_small / max(h, w))
    if s < 1.0:
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    else:
        small = img
    sigma = max(2.0, min(small.shape[:2]) / 12.0)  # v0.8.8 : /12 (au lieu de /8) — fond un peu plus « serré » pour capturer la lueur d'horizon, tout en lissant étoiles et Voie lactée
    glob = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if sky is not None:
        sk = cv2.resize(np.clip(sky.astype(np.float32), 0.0, 1.0), (small.shape[1], small.shape[0]),
                        interpolation=cv2.INTER_AREA)
        er = max(1, int(round(sigma / 2.0)))
        sk_pure = cv2.erode(sk, np.ones((3, 3), np.uint8), iterations=er)  # v0.8.5 : ciel pur uniquement
        if float(sk_pure.sum()) < 0.2 * float(sk.sum()) + 1e-6:
            sk_pure = sk  # ciel trop fin : on garde tout le ciel
        den = cv2.GaussianBlur(sk_pure, (0, 0), sigmaX=sigma, sigmaY=sigma)
        num = cv2.GaussianBlur(small * sk_pure[..., None], (0, 0), sigmaX=sigma, sigmaY=sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            masked = num / np.maximum(den, 1e-6)[..., None]
        bg_small = np.where((den > 0.05)[..., None], masked, glob)
    else:
        bg_small = glob
    if s < 1.0:
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bg = bg_small
    if squeeze:
        bg = bg[..., 0]
    return bg.astype(np.float32)


def synthetic_flat_correct(
    img: np.ndarray,
    sky: np.ndarray | None = None,
    strength: float = 1.0,
    sky_only: bool = True,
    target_pct: float = 10.0,
    feather: float | None = None,
    debug_cb=None,
) -> np.ndarray:
    """Soustrait un fond synthétique lisse (méthode par défaut).

    img     : float32 HxWxC, linéaire (0..65535 ou 0..1).
    sky     : poids flottants HxW (1 = ciel, 0 = avant-plan) ou None.
    strength: 0..1 (1 = correction complète).
    sky_only: si True et sky fourni, la correction n'est appliquée que sur le ciel
              (l'avant-plan est rendu tel quel, fondu sur le masque).
    target_pct : percentile du fond (sur le ciel) conservé comme niveau de fond cible.
    """
    img = img.astype(np.float32)
    bg = _synthetic_background(img, sky=sky)  # v0.8.3 : fond estimé sur le ciel uniquement
    out = img.copy()
    if sky is not None:
        sky_bool = sky > 0.5
        if not sky_bool.any():
            sky_bool = np.ones(img.shape[:2], dtype=bool)
    else:
        sky_bool = np.ones(img.shape[:2], dtype=bool)
    targets = []
    for c in range(img.shape[2]):
        t = float(np.percentile(bg[..., c][sky_bool], target_pct))
        targets.append(t)
        out[..., c] = img[..., c] - strength * (bg[..., c] - t)
    out = np.maximum(out, 0.0)
    if sky is not None and sky_only:
        wgt = _feather_weight(sky, img.shape, feather)[..., None]
        out = out * wgt + img * (1.0 - wgt)
    if debug_cb:
        debug_cb(
            "PL synthetic flat : cible ciel p{:g} = ".format(target_pct)
            + ", ".join(f"{t:.0f}" for t in targets)
            + f" (niveaux 0..{max(1.0, float(img.max())):.0f})"
        )
    return out


def _gaussian_kernel(sigma: float = 1.0, truncate: float = 4.0):
    ksize = round(sigma * truncate) - 1 if round(sigma * truncate) % 2 == 0 else round(sigma * truncate)
    return (ksize, ksize)


def _feather_weight(sky: np.ndarray, shape, feather: float | None = None) -> np.ndarray:
    """Poids de fusion du masque ciel, LISSÉ par flou gaussien (anti-halos).

    v0.8.2 : quand on ne corrige que le ciel (sky_only), un masque dur 0/1 crée un
    artefact visible au bord du masque peint par l'utilisateur (halo autour de
    l'avant-plan, cf. aperçu) — on lisse donc la transition.
    v0.8.5 : le poids est d'abord DILATÉ de sigma/2 avant le flou, ainsi la correction
    s'applique à fond CÔTÉ CIEL jusqu'au bord du masque (w ~ 1) et la transition se fait
    côté avant-plan — supprime le liseré clair collé au bord de l'avant-plan.
    v0.8.6 : dilatation portée à 1,5*sigma — avec un lissage symétrique le poids ne vaut
    qu'environ 0,7 au bord exact côté ciel, ce qui laissait ~30 % du gradient résiduel
    collé à l'avant-plan (halo persistant) ; à 1,5*sigma le bord ciel reçoit ~93 % de la
    correction et la transition est presque entièrement côté avant-plan.
    feather=None -> sigma auto = min(h, w) / 96 (v0.8.3, /128 en v0.8.2) ; feather=0 -> masque dur (sans
    lissage ni dilatation) ; sinon sigma = feather en pixels.
    """
    wgt = np.clip(np.asarray(sky, dtype=np.float32), 0.0, 1.0)
    sigma = max(2.0, min(shape[:2]) / 96.0) if feather is None else float(feather)
    if sigma > 0:
        grow = max(1, int(round(sigma * 1.5)))  # v0.8.6 : 1,5*sigma (0,5*sigma en v0.8.5)
        wgt = cv2.dilate(wgt, np.ones((3, 3), np.uint8), iterations=grow)  # le ciel « déborde » côté avant-plan
        wgt = cv2.GaussianBlur(wgt, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return wgt


def graxpert_correct(
    img: np.ndarray,
    sky: np.ndarray | None = None,
    strength: float = 1.0,
    sky_only: bool = True,
    smoothing: float = 0.1,
    feather: float | None = None,
    model_path: str | None = None,
    progress_cb=None,
    debug_cb=None,
) -> np.ndarray:
    """Correction PL par le modèle ONNX GraXpert (background extraction 1.0.1).

    Prétraitement identique à GraXpert (graxpert/background_extraction.py, branche AI) :
    resize 240x240, pad edge 8 -> 256x256, normalisation (x - médiane) / MAD * 0.04,
    clip [-1, 1], entrée NHWC "gen_input_image" ; le fond extrait est inversé, lissé
    (smoothing * 20, puis sigma 3), redimensionné, puis SOUSTRAIT en gardant le niveau
    moyen du fond (la PL est additive).
    """
    import onnxruntime as ort  # ImportError -> message clair côté appelant

    global _graxpert_session, _graxpert_path

    was_u16 = float(np.max(img)) > 1.5
    img01 = img.astype(np.float32)
    if was_u16:
        img01 = img01 / 65535.0

    path = ensure_graxpert_model(model_path, progress_cb=progress_cb)
    providers = [
        p
        for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
        if p in ort.get_available_providers()
    ] or ["CPUExecutionProvider"]
    if _graxpert_session is None or _graxpert_path != path:
        _graxpert_session = ort.InferenceSession(path, providers=providers)
        _graxpert_path = path
    if debug_cb:
        debug_cb(f"GraXpert : modèle {os.path.basename(path)}, providers {providers}")

    h, w = img01.shape[:2]
    c = img01.shape[2] if img01.ndim == 3 else 1
    padding = 8
    shrink = cv2.resize(img01, (256 - 2 * padding, 256 - 2 * padding), interpolation=cv2.INTER_LINEAR)
    if shrink.ndim == 2:
        shrink = shrink[..., None]
    shrink = np.pad(shrink, ((padding, padding), (padding, padding), (0, 0)), mode="edge")
    median = np.array([float(np.median(shrink[..., ch])) for ch in range(c)], dtype=np.float32)
    mad = np.array(
        [max(float(np.median(np.abs(shrink[..., ch] - median[ch]))), 1e-6) for ch in range(c)],
        dtype=np.float32,
    )
    norm = np.clip((shrink - median) / mad * 0.04, -1.0, 1.0)
    if c == 1:
        norm = np.repeat(norm, 3, axis=2)
    out = _graxpert_session.run(None, {"gen_input_image": np.expand_dims(norm, axis=0)})[0][0]
    bg = out / 0.04 * mad + median  # (256, 256, C), même échelle que img01
    if smoothing:
        sigma = smoothing * 20.0
        bg = cv2.GaussianBlur(bg, _gaussian_kernel(sigma), sigmaX=sigma, sigmaY=sigma)
    if padding:
        bg = bg[padding:-padding, padding:-padding, :]
    bg = cv2.GaussianBlur(bg, _gaussian_kernel(3.0), sigmaX=3.0, sigmaY=3.0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    if c == 1:
        bg = bg[..., :1]

    mean_bg = float(bg.mean())
    corrected = img01 - strength * (bg - mean_bg)
    corrected = np.maximum(corrected, 0.0).astype(np.float32)
    result = corrected * 65535.0 if was_u16 else corrected

    if sky is not None and sky_only:
        wgt = _feather_weight(sky, img.shape, feather)
        if result.ndim == 3:
            wgt = wgt[..., None]
        base = img.astype(np.float32)
        result = result * wgt + base * (1.0 - wgt)
    if debug_cb:
        debug_cb(
            f"GraXpert : fond extrait min={float(bg.min()):.4f} max={float(bg.max()):.4f} "
            f"(échelle 0..1), soustraction x{strength:.2f}"
        )
    return result


def correct_light_pollution(
    img: np.ndarray,
    method: str = "synthetic",
    strength: float = 1.0,
    sky: np.ndarray | None = None,
    sky_only: bool = True,
    smoothing: float = 0.1,
    feather: float | None = None,
    model_path: str | None = None,
    progress_cb=None,
    debug_cb=None,
) -> np.ndarray:
    """Point d'entrée : corrige le dégradé de pollution lumineuse (linéaire, post-stack).

    method  : "synthetic" (défaut) ou "graxpert" (ML ONNX).
    feather : sigma du lissage de la transition au bord du masque ciel (None = auto,
              0 = masque dur). Évite les halos quand sky_only=True.
    Retourne une image float32 de même échelle que l'entrée.
    """
    if method == "graxpert":
        return graxpert_correct(
            img,
            sky=sky,
            strength=strength,
            sky_only=sky_only,
            smoothing=smoothing,
            feather=feather,
            model_path=model_path,
            progress_cb=progress_cb,
            debug_cb=debug_cb,
        )
    return synthetic_flat_correct(
        img, sky=sky, strength=strength, sky_only=sky_only, feather=feather, debug_cb=debug_cb
    )
