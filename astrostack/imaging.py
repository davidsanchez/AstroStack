"""Petites fonctions d'image sans dépendance Qt (tonemapping pour aperçus, etc.)."""
from __future__ import annotations

import numpy as np


def stretch_u16(img: np.ndarray) -> np.ndarray:
    """float32 0..65535 -> uint16 0..65535 : même étirement que tonemap_u16 (percentiles + gamma 1/2.2),
    mais en 16 bits pour garder la précision. Sert au TIFF « étiré » destiné à l'affichage."""
    x = np.asarray(img, dtype=np.float32)
    x = np.clip(x / 65535.0, 0.0, 1.0)
    lo, hi = np.percentile(x, 0.1), np.percentile(x, 99.8)
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    x = x ** (1.0 / 2.2)
    return (x * 65535.0).round().astype(np.uint16)


def tonemap_u16(img: np.ndarray) -> np.ndarray:
    """float32/uint16 0..65535 -> uint8 HxWx3 (étirement percentiles + gamma 1/2.2)."""
    x = np.asarray(img, dtype=np.float32)
    x = np.clip(x / 65535.0, 0.0, 1.0)
    lo, hi = np.percentile(x, 0.1), np.percentile(x, 99.8)
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    x = x ** (1.0 / 2.2)
    return (x * 255.0).round().astype(np.uint8)
