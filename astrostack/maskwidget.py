"""Éditeur de masque d'avant-plan : on peint l'avant-plan avec un pinceau (comme dans
un outil de retouche photo), sur l'aperçu de l'image de référence.

Le masque vaut 255 là où l'utilisateur a peint (avant-plan, qui restera fixe),
0 ailleurs (ciel, qui sera recalé pour aligner les étoiles).
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .zoomview import ZoomMixin


class MaskEditor(ZoomMixin, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_zoom()
        self._pos_w: tuple[float, float] | None = None  # position du pinceau (pixels du widget)
        self.setMouseTracking(True)
        self._qimg: QImage | None = None
        self._mask: np.ndarray | None = None  # uint8 HxW, 255 = avant-plan peint
        self._margin_cache: np.ndarray | None = None  # extension (dilatation) affichée en orange
        # calques RGBA pré-calculés (rouge, bleu, orange), mis à jour par zone au pinceau
        self._rgba: dict[str, np.ndarray | None] = {"red": None, "blue": None, "orange": None}
        self._fit: np.ndarray | None = None   # uint8 HxW, 255 = zone exclue du CALCUL PL (nuages)
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
        self._margin_cache = None
        self._invalidate("red", "blue", "orange")
        self.refresh_zoom()

    def _content_size(self) -> tuple[int, int]:
        if self._qimg is None:
            return (0, 0)
        return (self._qimg.width(), self._qimg.height())

    def _invalidate(self, *layers: str) -> None:
        """Oublie les calques RGBA (reconstruits au prochain affichage)."""
        for k in layers:
            self._rgba[k] = None

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
        self._margin_cache = None
        self._invalidate("red", "orange")
        self.update()

    def get_fit_mask(self) -> np.ndarray | None:
        """Zone exclue du calcul PL (255 = exclue), ou None si rien n'est peint."""
        if self._fit is None or not self._fit.any():
            return None
        return self._fit.copy()

    def clear_fit_mask(self) -> None:
        if self._fit is not None:
            self._fit[:] = 0
            self._invalidate("blue")
            self.update()

    def has_image(self) -> bool:
        return self._qimg is not None

    def get_mask(self) -> np.ndarray | None:
        return None if self._mask is None else self._mask.copy()

    def clear_mask(self) -> None:
        if self._mask is not None:
            self._mask[:] = 0
            self._margin_cache = None
            self._invalidate("red", "orange")
            self.update()

    def invert_mask(self) -> None:
        if self._mask is not None:
            self._mask = np.where(self._mask > 0, 0, 255).astype(np.uint8)
            self._margin_cache = None
            self._invalidate("red", "orange")
            self.update()

    # --- peinture ---
    def _img_pos(self, ev) -> tuple[int, int] | None:
        """Coordonnées image (pixels du masque) sous le curseur, en tenant compte du zoom."""
        if self._qimg is None or self.width() <= 0:
            return None
        iw, ih = self._qimg.width(), self._qimg.height()
        x, y = ev.position().x(), ev.position().y()
        ix = int(x * iw / self.width())
        iy = int(y * ih / self.height())
        if 0 <= ix < iw and 0 <= iy < ih and 0 <= x < self.width() and 0 <= y < self.height():
            return ix, iy
        return None

    def _brush_px(self) -> float:
        """Rayon du pinceau en pixels du widget (il suit le zoom)."""
        if self._qimg is None or self.width() <= 0:
            return max(1.0, self.brush_size / 2)
        return max(1.0, self.brush_size / 2) * self.width() / self._qimg.width()

    def _paint(self, ev) -> None:
        pos = self._img_pos(ev)
        if pos is None or self._mask is None:
            return
        x, y = pos
        target = self._fit if self.paint_fit else self._mask
        if target is None:
            return
        # seule la boîte autour du pinceau est mise à jour (pas toute l'image)
        r = max(1, self.brush_size // 2)
        H, W = target.shape
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        sub = target[y0:y1, x0:x1]
        sub[m] = 0 if self.erase else 255
        # calque déjà construit : on ne met à jour que la boîte du pinceau
        layer = "blue" if self.paint_fit else "red"
        rgba = self._rgba[layer]
        if rgba is not None:
            rgba[y0:y1, x0:x1, 3] = (sub > 0).astype(np.uint8) * 110
        self.update()

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._painting = True
            self._paint(ev)

    def mouseMoveEvent(self, ev) -> None:
        self._pos = self._img_pos(ev)
        self._pos_w = (ev.position().x(), ev.position().y()) if self._pos is not None else None
        if self._painting and (ev.buttons() & Qt.LeftButton):
            self._paint(ev)
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._painting = False
        # la marge (dilatation) n'est recalculée qu'à la fin du trait, pas à chaque point
        self._margin_cache = None
        self._invalidate("orange")
        self.update()

    # --- dessin ---
    def _margin(self) -> np.ndarray:
        """v1.2.0 : pixels ajoutés par la dilatation de marge (même règle que le pipeline)."""
        if self._margin_cache is None:
            import cv2
            base = (self._mask > 0).astype(np.uint8)
            k = max(3, int(round(min(base.shape) * 0.005))) | 1
            dil = cv2.dilate(base, np.ones((k, k), np.uint8))
            self._margin_cache = ((dil > 0) & (base == 0)).astype(np.uint8)
        return self._margin_cache

    def _layer(self, kind: str) -> np.ndarray | None:
        """Calque RGBA (h, w, 4) du rouge (masque), du bleu (zone PL) ou de l'orange (marge)."""
        if self._rgba[kind] is not None:
            return self._rgba[kind]
        if kind == "red":
            if self._mask is None or self._qimg is None:
                return None
            a = np.zeros(self._mask.shape + (4,), np.uint8)
            a[..., 0] = 255
            a[..., 3] = (self._mask > 0).astype(np.uint8) * 110
        elif kind == "blue":
            if self._fit is None or self._qimg is None:
                return None
            a = np.zeros(self._fit.shape + (4,), np.uint8)
            a[..., 1] = 140  # bleu : zone exclue du calcul PL seulement
            a[..., 2] = 255
            a[..., 3] = (self._fit > 0).astype(np.uint8) * 110
        else:  # orange : extension par dilatation (marge)
            if self._mask is None or self._qimg is None or not self._mask.any():
                return None
            mg = self._margin()
            a = np.zeros(self._mask.shape + (4,), np.uint8)
            a[..., 0] = 255
            a[..., 1] = 140
            a[..., 3] = mg * 150
        self._rgba[kind] = a
        return a

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        full = QRectF(0, 0, self.width(), self.height())
        if self._qimg is not None:
            p.drawImage(full, self._qimg)
        else:
            p.fillRect(self.rect(), QColor(30, 30, 30))
        if self._qimg is not None:
            for kind in ("red", "orange", "blue"):
                a = self._layer(kind)
                if a is None or (kind == "blue" and not self._fit.any()):
                    continue
                h, w = a.shape[:2]
                ov = QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888)  # partage la mémoire du calque
                p.drawImage(full, ov)
        if self._pos_w is not None:
            r = self._brush_px()
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(*self._pos_w), r, r)
        p.end()
