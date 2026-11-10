"""Zoom et centrage partagés par l'aperçu de référence (masque) et le résultat.

Par défaut le contenu est « ajusté à la fenêtre » (il tient entièrement dans la zone
visible et est centré). Ctrl + molette, ou les boutons de la barre de zoom, passent
en facteur absolu ; « Ajuster » revient à l'ajustement automatique.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QScrollArea, QWidget

ZOOM_MIN = 0.05
ZOOM_MAX = 8.0
ZOOM_STEP = 1.25


class ZoomMixin:
    """Ajoute zoom + ajustement à un QWidget de contenu placé dans un QScrollArea.

    Sous-classe : fournir `_content_size() -> (w, h)`. Le widget prend la taille du
    contenu à l'échelle courante ; le QScrollArea (alignement centré) le centre tant
    qu'il est plus petit que la zone visible."""

    def _init_zoom(self) -> None:
        self._zoom: float | None = None  # None = ajusté à la fenêtre

    def _content_size(self) -> tuple[int, int]:  # pragma: no cover - à surcharger
        raise NotImplementedError

    def _fit_scale(self) -> float:
        w, h = self._content_size()
        parent = self.parentWidget()
        vp = parent.size() if parent is not None else self.size()
        if w <= 0 or h <= 0 or vp.width() < 20 or vp.height() < 20:
            return 1.0
        return max(ZOOM_MIN, min((vp.width() - 4) / w, (vp.height() - 4) / h))

    def zoom_is_fit(self) -> bool:
        return self._zoom is None

    def zoom_scale(self) -> float:
        return self._fit_scale() if self._zoom is None else self._zoom

    def zoom_by(self, factor: float) -> None:
        cur = self.zoom_scale()
        self._zoom = min(ZOOM_MAX, max(ZOOM_MIN, cur * factor))
        self.refresh_zoom()

    def zoom_fit(self) -> None:
        self._zoom = None
        self.refresh_zoom()

    def zoom_100(self) -> None:
        self._zoom = 1.0
        self.refresh_zoom()

    def refresh_zoom(self) -> None:
        w, h = self._content_size()
        if w <= 0 or h <= 0:
            return
        s = self.zoom_scale()
        self.setFixedSize(max(1, round(w * s)), max(1, round(h * s)))
        self.update()


class ZoomScroll(QScrollArea):
    """Zone de défilement centrant son contenu ; signale les changements de taille."""

    resized = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        w = self.widget()
        if isinstance(w, ZoomMixin) and w.zoom_is_fit():
            w.refresh_zoom()
        self.resized.emit()


class ResultCanvas(ZoomMixin, QWidget):
    """Affiche le résultat (QPixmap pleine résolution) à l'échelle du zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_zoom()
        self._pix: QPixmap | None = None
        self._cache: tuple[float, QPixmap] | None = None
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setFixedSize(320, 200)

    def set_pixmap(self, pix: QPixmap | None) -> None:
        self._pix = pix if (pix is not None and not pix.isNull()) else None
        self._cache = None
        self.refresh_zoom()

    def _content_size(self) -> tuple[int, int]:
        if self._pix is None:
            return (0, 0)
        return (self._pix.width(), self._pix.height())

    def refresh_zoom(self) -> None:
        if self._pix is None:
            parent = self.parentWidget()
            vp = parent.size() if parent is not None else self.size()
            self.setFixedSize(max(1, vp.width()), max(1, vp.height()))
            self.update()
            return
        super().refresh_zoom()

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        if self._pix is None:
            p.setPen(QColor(150, 150, 160))
            p.drawText(self.rect(), Qt.AlignCenter, "Le résultat apparaîtra ici.")
        else:
            w, h = self.width(), self.height()
            if self._cache is None or self._cache[1].width() != w or self._cache[1].height() != h:
                self._cache = (self.zoom_scale(), self._pix.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            p.drawPixmap(QRectF(0, 0, w, h).toRect(), self._cache[1])
        p.end()
