"""Masque ML (ONNX) — comme le "ML generated mask" de SharpStacker, mais éditable à la main.

Modèle par défaut : SegFormer-B0 fine-tuné sur ADE20K (512x512), en ONNX quantifié
(~14 Mo), téléchargé depuis Hugging Face au premier usage puis mis en cache dans
~/.astrostack/. La classe "sky" d'ADE20K est l'index 2.

v0.4.0 : au lieu d'un argmax binaire remonté en NEAREST (qui donnait des "gros carrés"
de 512 px), on prend la PROBABILITÉ softmax du canal ciel, remontée à la taille de
l'image en INTER_CUBIC puis lissée (flou gaussien) — bords nets et continus, toujours
retouchable au pinceau dans l'éditeur.

Personnalisable via les variables d'environnement :
  ASTROSTACK_ONNX_MODEL  : chemin local du modèle .onnx
  ASTROSTACK_ONNX_URL    : URL de téléchargement du modèle
  ASTROSTACK_SKY_CLASS   : index de la classe "ciel" (défaut 2, ADE20K)
Un modèle à sortie sigmoïde (1 canal) est aussi supporté.

v0.7.6 : correction du layout des logits (canaux en dernier) + repli luminance quand le
modèle ne reconnaît pas le ciel nocturne (sinon toute l'image partait en avant-plan),
avec diagnostics via debug_cb.
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_MODEL_URL = (
    "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx"
)
DEFAULT_MODEL_PATH = os.path.join(os.path.expanduser("~"), ".astrostack", "segformer-b0-ade20k.onnx")
DEFAULT_SKY_CLASS = 2  # ADE20K : 0=wall, 1=building, 2=sky, ...
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None
_session_path: str | None = None


def model_path() -> str:
    return os.environ.get("ASTROSTACK_ONNX_MODEL") or DEFAULT_MODEL_PATH


def sky_class() -> int:
    return int(os.environ.get("ASTROSTACK_SKY_CLASS", DEFAULT_SKY_CLASS))


def ensure_model(path: str | None = None, url: str | None = None, progress_cb=None) -> str:
    """Retourne le chemin du modèle, en le téléchargeant si absent."""
    path = path or model_path()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = url or os.environ.get("ASTROSTACK_ONNX_URL") or DEFAULT_MODEL_URL
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if progress_cb:
        progress_cb(f"Téléchargement du modèle ONNX ({url})…")
    import urllib.request

    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """v0.8.11 — seuil d'Otsu sur des valeurs normalisées 0..1 (histogramme numpy, sans cv2)."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0).ravel()
    hist, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
    var_between = np.nan_to_num(var_between)
    return float(centers[int(np.argmax(var_between))])


def luma_u8(img_u8_rgb: np.ndarray) -> np.ndarray:
    """v0.8.11 — luminance Rec.709 normalisée 0..1 (float32) d'une image uint8 RGB."""
    a = img_u8_rgb.astype(np.float32)
    return (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]) / 255.0


def predict_sky_mask(img_u8_rgb: np.ndarray, path: str | None = None, input_size: int = 512,
                     sky_cls: int | None = None, top_gradient: bool = False,
                     top_boost: float = 0.5, proba_blur: float = 1.0) -> np.ndarray:
    """img_u8_rgb : HxWx3 uint8. Retourne un masque float32 HxW continu, 1.0 = ciel, 0.0 = avant-plan.

    v0.4.0 : probabilité softmax du canal ciel (pas un argmax binaire), remontée en
    INTER_CUBIC à la taille de l'image puis lissée au gaussien -> plus de "blocs" 512 px.
    v0.7.8 : `sky_cls` permet de choisir l'index de la classe « ciel » (défaut : env
    ASTROSTACK_SKY_CLASS ou 2 = ADE20K sky).
    v0.7.9 : `top_gradient` applique un dégradé vertical à la proba ciel — le HAUT de
    l'image est favorisé comme ciel (boost max `top_boost` en haut, 0 en bas).
    """
    global _session, _session_path
    import cv2

    path = ensure_model(path)
    if _session is None or _session_path != path:
        import onnxruntime as ort  # dépendance optionnelle (pip install onnxruntime)

        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _session_path = path

    k = sky_cls if sky_cls is not None else sky_class()

    inp = _session.get_inputs()[0].name
    x = cv2.resize(img_u8_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x.transpose(2, 0, 1)[None].astype(np.float32)  # 1x3xHxW
    out = _session.run(None, {inp: x})[0]

    if out.ndim == 4 and out.shape[1] > 1:  # logits multi-classes -> probabilité softmax du canal ciel
        logits = out[0].astype(np.float32)  # C x h x w (PyTorch) ou h x w x C (certains exports)
        if logits.ndim == 3 and logits.shape[-1] >= logits.shape[0]:
            logits = logits.transpose(2, 0, 1)  # canaux en dernier -> canaux en premier
        proba = _softmax(logits)  # softmax sur l'axe des classes
        sky = proba[min(k, proba.shape[0] - 1)]
    else:  # sigmoïde / probabilité binaire
        p = out.astype(np.float32).reshape(out.shape[-2], out.shape[-1])
        sky = 1.0 / (1.0 + np.exp(-p)) if float(p.min()) < 0.0 or float(p.max()) > 1.0 else p

    h, w = img_u8_rgb.shape[:2]
    sky = cv2.resize(sky.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    sky = np.clip(sky, 0.0, 1.0)
    if proba_blur > 0:  # v0.8.9 : proba_blur=0 -> AUCUN lissage de la proba (masque ONNX brut pour la PL)
        sky = cv2.GaussianBlur(sky, (0, 0), sigmaX=proba_blur, sigmaY=proba_blur)  # lissage fin des bords
    if top_gradient:
        # dégradé vertical : le haut de l'image est favorisé comme ciel
        yy = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
        sky = np.clip(sky * (1.0 + top_boost * yy), 0.0, 1.0)
    return sky.astype(np.float32)


def sky_to_foreground_mask(img_u8_rgb: np.ndarray, path: str | None = None, smooth: int = 3,
                           debug_cb=None, sky_cls: int | None = None,
                           top_gradient: bool = False, top_boost: float = 0.5,
                           grow: int = 0, proba_blur: float = 1.0, morph: bool = True,
                           lum_refine: bool = False, lum_thr: float | None = None,
                           lum_add: float = 1.5, edge_smooth: int = 5,
                           drop_small: float = 0.0015,
                           lum_edge: bool = False, edge_band: int = 40) -> np.ndarray:
    """Masque d'avant-plan uint8 (255 = avant-plan), prêt pour MaskEditor.set_mask().

    Le masque soft (1 - proba_ciel) est seuillé à 0.5, nettoyé par morphologie (open/close
    3x3) puis reslendé légèrement -> bords propres et continus, éditable au pinceau.

    v0.8.3 : `grow` dilatait le masque d'avant-plan de N pixels partout
    (cv2.dilate, itérations=N) — marge de sécurité autour des objets fixes.
    v0.8.12 : `grow` vaut désormais 0 PAR DÉFAUT (supprimé sur le masque ML d'empilement :
    il rognait le ciel au bord des objets) ; le pipeline applique sa propre marge de
    sécurité à part.

    v0.8.9 : pour un masque ONNX 100 % BRUT (chemin correction PL), passer
    `top_gradient=False, grow=0, smooth=1, proba_blur=0, morph=False` : aucun dégradé,
    aucune dilatation, aucun lissage de proba, aucune morphologie — juste le seuil 0,5
    de la probabilité « ciel ».

    v0.7.6 : si le modèle ne détecte quasiment AUCUN ciel (proba moyenne < 0.02 — typique
    d'un ciel NOCTURNE, absent de la distribution ADE20K du modèle), on bascule sur un
    repli par luminance (en photo de nuit, l'avant-plan est sombre) au lieu de marquer
    toute l'image comme avant-plan. `debug_cb(msg)` reçoit les diagnostics.

    v0.8.11 : `lum_refine=True` affine le masque ONNX avec la LUMINANCE de l'image
    (utile pour l'EMPILEMENT : avant-plan sombre vs ciel lumineux) — seuil `lum_thr`
    (0..255, None = auto par Otsu) : pixels SOMBRES forcés en avant-plan, et grandes
    zones TRÈS lumineuses (seuil × `lum_add`, nettoyées par ouverture morphologique pour
    éliminer les petites sources ponctuelles : fenêtres, étoiles isolées) basculées en
    CIEL — typiquement la lueur d'horizon, que le modèle ONNX classe souvent « avant-plan ».
    v0.8.12 : l'affinage luminance est ANCRÉ sur le masque ONNX de base — un pixel sombre
    ne devient avant-plan que s'il est RELIÉ (dilatation ~2,5 % du petit côté) au blob
    avant-plan ONNX, et une zone lumineuse ne devient ciel que si elle est reliée au ciel
    ONNX : plus d'îlots sans lien (ex. coins vignettés sombres en haut de l'image).
    v0.8.12 : `edge_smooth` (défaut 5) = lissage médian du masque binaire qui efface les
    bords en escalier venus du sur-échantillonnage 512 px de la proba ONNX (granularité) ;
    `drop_small` (défaut 0,15 % de l'image, min 150 px) = supprime les petites composantes
    connexes d'avant-plan isolées (bruit, artefacts).
    v0.8.14 : `lum_edge=True` recale le BORD du masque sur la luminance pleine résolution :
    dans une bande de ±`edge_band` px autour de la frontière ONNX, la classe devient
    purement `lum <= seuil` (seuil Otsu auto ou `lum_thr`) — les fines branches sombres
    sur ciel lumineux sont alors détectées nativement à la résolution de l'image, sans
    toucher au ciel loin du bord (pas d'artefacts). L'ONNX garde la forme globale.
    """
    import cv2

    def _dbg(msg: str) -> None:
        if debug_cb:
            debug_cb(msg)

    k = sky_cls if sky_cls is not None else sky_class()
    _dbg(f"ML : classe « ciel » = {k} (ADE20K : 2 = sky)")
    if top_gradient:
        _dbg(f"ML : dégradé « haut = ciel » activé (boost {top_boost})")
    sky = predict_sky_mask(img_u8_rgb, path=path, sky_cls=k,
                           top_gradient=top_gradient, top_boost=top_boost,
                           proba_blur=proba_blur)
    sky_mean = float(sky.mean())
    _dbg(f"ML : proba « ciel » moyenne = {sky_mean:.3f} (max = {float(sky.max()):.3f})")
    if sky_mean < 0.02:
        _dbg("ML : aucun ciel reconnu (ciel nocturne, hors distribution ADE20K) — "
             "repli luminance : pixels sombres = avant-plan.")
        lum = (0.2126 * img_u8_rgb[..., 0] + 0.7152 * img_u8_rgb[..., 1]
               + 0.0722 * img_u8_rgb[..., 2]) / 255.0
        thr = 40.0 / 255.0  # même seuil que le bouton « Auto » (40/255)
        fg = np.clip((thr - lum) / thr, 0.0, 1.0)
    else:
        fg = 1.0 - sky
        if lum_refine or lum_edge:  # v0.8.14 : luminance + seuil calculés une seule fois si besoin
            lum = luma_u8(img_u8_rgb)
            t = (lum_thr / 255.0) if lum_thr else otsu_threshold(lum)
        if lum_refine:  # v0.8.11/0.8.12 : affinage luminance ANCRÉ sur le masque ONNX de base
            hi = min(1.0, t * lum_add)
            h_, w_ = lum.shape
            r = max(15, min(h_, w_) // 40)  # rayon de « lien » ~2,5 % du petit côté
            kl = np.ones((r, r), np.uint8)
            fg_bin = (fg > 0.5).astype(np.uint8)
            link_fg = cv2.dilate(fg_bin, kl) > 0       # relié à l'avant-plan ONNX
            link_sky = cv2.dilate(1 - fg_bin, kl) > 0  # relié au ciel ONNX
            dark = ((lum <= t) & link_fg).astype(np.float32)  # sombre ET relié -> avant-plan
            fg = np.maximum(fg, dark)
            if hi < 1.0:
                bright = (lum > hi).astype(np.uint8)
                bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                bright = ((bright > 0) & link_sky).astype(np.float32)  # lumineux ET relié au ciel -> ciel
                fg = np.minimum(fg, 1.0 - bright)
            _dbg(f"ML : affinage luminance ancré sur l'ONNX — seuil {'manuel' if lum_thr else 'Otsu'} = {t:.3f} "
                 f"({t * 255:.0f}/255), lien {r} px, zones > {hi:.3f} reliées au ciel -> ciel ; "
                 f"ciel après affinage = {float((fg < 0.5).mean()):.0%} de l'image")
        if lum_edge:  # v0.8.14 : recalage du BORD sur la luminance pleine résolution (bande ±edge_band px)
            fg_bin = (fg > 0.5).astype(np.uint8)
            B = max(1, int(edge_band))
            kB = np.ones((2 * B + 1, 2 * B + 1), np.uint8)
            band = (cv2.dilate(fg_bin, kB) > 0) & (cv2.erode(fg_bin, kB) == 0)  # frontière ±B px
            fg = np.where(band, (lum <= t).astype(np.float32), fg)
            _dbg(f"ML : bord recalé sur la luminance dans une bande de ±{B} px "
                 f"(seuil {t:.3f} = {t * 255:.0f}/255) — détails fins au bord, ciel loin du bord intact")
    b = (fg > 0.5).astype(np.uint8)
    if drop_small > 0:  # v0.8.12 : supprime les îlots d'avant-plan sans lien (artefacts, bruit)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        if n > 1:
            min_area = max(150.0, drop_small * b.size)
            dropped = 0
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    b[labels == i] = 0
                    dropped += 1
            if dropped:
                _dbg(f"ML : {dropped} îlot(s) d'avant-plan < {min_area:.0f} px supprimés (sans lien avec le gros blob)")
    if edge_smooth > 1:  # v0.8.12 : efface les bords en escalier (granularité, upsample 512 px)
        b = cv2.medianBlur(b, edge_smooth | 1)
    k = np.ones((3, 3), np.uint8)
    if morph:  # v0.8.9 : morph=False -> aucun nettoyage morphologique (masque brut)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    if grow > 0:  # v0.8.3 : avant-plan agrandi de `grow` px partout
        b = cv2.dilate(b, k, iterations=grow)
        _dbg(f"ML : masque avant-plan dilaté de {grow} px")
    if smooth > 1:
        kk = smooth | 1
        fg = cv2.GaussianBlur(b.astype(np.float32), (kk, kk), 0)
    else:
        fg = b.astype(np.float32)
    return (np.clip(fg, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def foreground_mask_from_linear(img_f32: np.ndarray, max_side: int = 1200, **kw) -> np.ndarray:
    """v0.8.5 — Masque d'avant-plan uint8 (255 = avant-plan) à la TAILLE de `img_f32`,
    depuis une image linéaire float32 (0..65535) : downscale <= max_side px, tonemap,
    inférence ONNX (sky_to_foreground_mask), puis re-upscale. kwargs -> sky_to_foreground_mask
    (debug_cb, top_gradient, top_boost, grow…)."""
    import cv2

    from .imaging import tonemap_u16

    h, w = img_f32.shape[:2]
    s = min(1.0, max_side / max(h, w))
    r = img_f32 if s >= 1.0 else cv2.resize(
        img_f32, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
    )
    r = r.astype(np.float32)
    if r.ndim == 2:
        r = np.stack([r] * 3, axis=-1)
    fg = sky_to_foreground_mask(tonemap_u16(r), **kw)
    if s < 1.0:
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    return fg
