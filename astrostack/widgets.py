"""Widgets réutilisables de l'interface.

- CollapsibleBox : section repliable (cadre + case à cocher d'en-tête).
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QGroupBox, QVBoxLayout, QWidget


class CollapsibleBox(QGroupBox):
    """Section repliable : look d'avant (cadre de groupbox + case à cocher en titre),
    mais avec une QCheckBox d'en-tête réellement cliquable — un QGroupBox checkable
    stylé en QSS ne répondait plus au clic (bug Qt connu). Le contenu se replie
    SUR LA HAUTEUR."""

    def __init__(self, title: str, checked: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("CollapsibleBox")
        self._content = QWidget()
        self._lay = QVBoxLayout(self._content)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(3)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 1, 4, 4)
        outer.setSpacing(2)
        self._title = title
        self.chk = QCheckBox()
        self.chk.setObjectName("cbHeader")
        self.chk.setChecked(checked)
        self.chk.setCursor(Qt.PointingHandCursor)
        self.chk.toggled.connect(self._content.setVisible)
        self.chk.toggled.connect(self._on_toggled_arrow)
        self._on_toggled_arrow(checked)
        outer.addWidget(self.chk)
        outer.addWidget(self._content)
        self._content.setVisible(checked)

    def addWidget(self, w) -> None:
        self._lay.addWidget(w)

    def addLayout(self, l) -> None:
        self._lay.addLayout(l)

    def _on_toggled_arrow(self, checked: bool) -> None:
        """v0.8.8 : petite flèche dans le titre — ▾ (ouverte) / ▸ (fermée) —
        à la place de la case à cocher (indicateur masqué en QSS)."""
        self.chk.setText(("▼ " if checked else "▶ ") + self._title)
