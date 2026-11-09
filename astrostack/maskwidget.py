"""Éditeur de masque d'avant-plan : on peint l'avant-plan avec un pinceau (comme dans
un outil de retouche photo), sur l'aperçu de l'image de référence.

Le masque vaut 255 là où l'utilisateur a peint (avant-plan, qui restera fixe),
0 ailleurs (ciel, qui sera recalé pour aligner les étoiles).
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .preview import qimage_to_u8_rgb


class MaskEditor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._qimg: QImage | None = None
        self._mask: np.ndarray | None = None  # uint8 HxW, 255 = avant-plan peint
        self._fit: np.ndarray | None = None   # v1.1.5 : uint8 HxW, 255 = zone exclue du CALCUL PL (nuages)
        self.paint_fit = False                # True : le pinceau peint la zone d'exclusion PL
        self.brush_size = 40  # diamètre en pixels de l'aperçu
        self.erase = False
        self._pos: tuple[int, int] | None = None
        self._painting = False

    # --- contenu ---
    def set_image(self, qimg: QImage) -> None:
        self._qimg = qimg
        self._mask = np.zeros((qimg.height(), qimg.width()), dtype=np.uint8)
        self._fit = np.zeros((qimg.height(), qimg.width()), dtype=np.uint8)
        self.setFixedSize(qimg.width(), qimg.height())
        self.update()

    def image(self) -> QImage | None:
        """QImage actuellement affichée (pour alimenter le masque ML)."""
        return self._qimg

    def set_mask(self, mask_u8: np.ndarray) -> None:
        """Pose un masque externe (ex. masque ML), redimensionné à l'aperçu si besoin."""
        if self._qimg is None:
            return
        m = np.asarray(mask_u8)
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape != (self._qimg.height(), self._qimg.width()):
            import cv2

            m = cv2.resize(m, (self._qimg.width(), self._qimg.height()), interpolation=cv2.INTER_LINEAR)
        self._mask = m.astype(np.uint8)
        self.update()

    def get_fit_mask(self) -> np.ndarray | None:
        """Zone exclue du calcul PL (255 = exclue), ou None si rien n'est peint."""
        if self._fit is None or not self._fit.any():
            return None
        return self._fit.copy()

    def clear_fit_mask(self) -> None:
        if self._fit is not None:
            self._fit[:] = 0
            self.update()

    def has_image(self) -> bool:
        return self._qimg is not None

    def get_mask(self) -> np.ndarray | None:
        return None if self._mask is None else self._mask.copy()

    def clear_mask(self) -> None:
        if self._mask is not None:
            self._mask[:] = 0
            self.update()

    def invert_mask(self) -> None:
        if self._mask is not None:
            self._mask = np.where(self._mask > 0, 0, 255).astype(np.uint8)
            self.update()

    def auto_threshold(self, thr: int) -> None:
        """Masque auto heuristique : avant-plan = pixels plus sombres que `thr` (0..255).

        C'est un point de départ (équivalent manuel du masque ML de SharpStacker) ;
        l'utilisateur affine ensuite au pinceau."""
        if self._qimg is None:
            return
        arr = qimage_to_u8_rgb(self._qimg)
        lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]).astype(np.uint8)
        m = np.where(lum <= thr, 255, 0).astype(np.uint8)
        k = np.ones((5, 5), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        # bords adoucis (le pipeline floute de toute façon pour le fondu)
        mf = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)
        self._mask = np.where(mf > 127, 255, 0).astype(np.uint8)
        self.update()

    # --- peinture ---
    def _img_pos(self, ev) -> tuple[int, int] | None:
        if self._qimg is None:
            return None
        x, y = ev.position().x(), ev.position().y()
        if 0 <= x < self._qimg.width() and 0 <= y < self._qimg.height():
            return int(x), int(y)
        return None

    def _paint(self, ev) -> None:
        pos = self._img_pos(ev)
        if pos is None or self._mask is None:
            return
        x, y = pos
        target = self._fit if self.paint_fit else self._mask
        if target is None:
            return
        r = max(1, self.brush_size // 2)
        yy, xx = np.ogrid[: target.shape[0], : target.shape[1]]
        m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        target[m] = 0 if self.erase else 255
        self.update()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._painting = True
            self._paint(ev)

    def mouseMoveEvent(self, ev) -> None:
        self._pos = self._img_pos(ev)
        if self._painting and (ev.buttons() & Qt.LeftButton):
            self._paint(ev)
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._painting = False

    # --- dessin ---
    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        if self._qimg is not None:
            p.drawImage(0, 0, self._qimg)
        else:
            p.fillRect(self.rect(), QColor(30, 30, 30))
        if self._qimg is not None and self._mask is not None:
            h, w = self._mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = 255  # rouge
            rgba[..., 3] = (self._mask > 0).astype(np.uint8) * 110  # alpha
            overlay = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
            p.drawImage(0, 0, overlay)
        if self._qimg is not None and self._fit is not None and self._fit.any():
            h, w = self._fit.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = 0
            rgba[..., 1] = 140  # bleu : zone exclue du calcul PL seulement
            rgba[..., 2] = 255
            rgba[..., 3] = (self._fit > 0).astype(np.uint8) * 110
            overlay_fit = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
            p.drawImage(0, 0, overlay_fit)
        if self._pos is not None:
            r = max(1, self.brush_size // 2)
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(*self._pos), r, r)
        p.end()
