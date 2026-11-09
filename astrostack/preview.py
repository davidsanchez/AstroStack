"""Conversion numpy <-> QImage pour l'interface Qt."""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

from .imaging import tonemap_u16


def u8_rgb_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def u16_to_qimage(img: np.ndarray) -> QImage:
    return u8_rgb_to_qimage(tonemap_u16(img))


def qimage_to_u8_rgb(qimg: QImage) -> np.ndarray:
    img = qimg.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    buf = img.bits()
    arr = np.frombuffer(buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
    return arr.copy()
