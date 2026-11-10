"""Feuille de style sombre de l'interface (Qt).

Les couleurs sont celles du thème « sombre » (Fusion + QSS), identique sur macOS et Windows.
"""

DARK_QSS = """
QMainWindow, QWidget { background-color: #1b1b1f; color: #e8e8ec; }
QGroupBox { border: 1px solid #3a3a42; border-radius: 6px; margin-top: 12px; padding: 4px 6px 6px 6px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 4px; color: #9ecbff; }
QGroupBox::indicator { width: 14px; height: 14px; border: 1px solid #45454f; border-radius: 3px; background-color: #2e2e36; }
QGroupBox::indicator:checked { background-color: #4a90d9; border-color: #6db3f2; }
QGroupBox#CollapsibleBox { margin-top: 4px; padding: 2px 6px 6px 6px; background-color: #202027; }
QCheckBox#cbHeader { font-weight: 600; color: #9ecbff; font-size: 12px; spacing: 4px; }
QCheckBox#cbHeader::indicator { width: 0px; height: 0px; border: none; background: transparent; }  /* v0.8.8 : indicateur masqué, flèche ▸/▾ dans le titre */
QListWidget { background-color: #232329; border: 1px solid #3a3a42; border-radius: 8px; color: #e8e8ec; }
QPushButton { background-color: #2a2a33; border: 1px solid #3a3a46; border-radius: 8px; padding: 5px 12px; }
QPushButton:hover { background-color: #34343f; border-color: #4a90d9; }
QPushButton:pressed { background-color: #23232b; }
QPushButton:disabled { color: #6a6a74; background-color: #222229; }
QSlider::groove:horizontal { height: 6px; background: #34343f; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #4a90d9; border-radius: 3px; }
QSlider::handle:horizontal { background: #9ecbff; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
QListWidget::item { padding: 1px; }
QPushButton { background-color: #2e2e36; border: 1px solid #45454f; border-radius: 4px; padding: 4px 8px; color: #e8e8ec; }
QPushButton:hover { background-color: #3a3a44; }
QPushButton:pressed { background-color: #26262d; }
QPushButton:disabled { color: #6f6f7a; background-color: #26262c; border-color: #34343c; }
QCheckBox:disabled { color: #6f6f7a; }  /* v0.8.11 : contraste net entre actif (#e8e8ec) et grisé */
QLabel:disabled { color: #6f6f7a; }
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background-color: #222228; color: #6f6f7a; border-color: #34343c; }
QToolButton { background-color: transparent; border: none; color: #ff7b72; font-weight: bold; font-size: 13px; padding: 0 4px; }
QToolButton:hover { color: #ff5347; }
QPlainTextEdit { background-color: #141417; color: #b5e8b0; border: 1px solid #3a3a42; border-radius: 4px; }
QProgressBar { background-color: #232329; border: 1px solid #3a3a42; border-radius: 4px; text-align: center; color: #e8e8ec; }
QProgressBar::chunk { background-color: #4a90d9; border-radius: 3px; }
QComboBox, QSpinBox, QDoubleSpinBox { background-color: #2e2e36; color: #e8e8ec; border: 1px solid #45454f; border-radius: 4px; padding: 2px 4px; }
QComboBox QAbstractItemView { background-color: #2e2e36; color: #e8e8ec; selection-background-color: #4a90d9; }
QCheckBox { color: #e8e8ec; spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #45454f; border-radius: 3px; background-color: #2e2e36; }
QCheckBox::indicator:hover { border-color: #4a90d9; }
QCheckBox::indicator:checked { background-color: #4a90d9; border-color: #6db3f2; }
QCheckBox::indicator:disabled { background-color: #232329; border-color: #3a3a42; }
QLabel { color: #e8e8ec; }
QTabWidget::pane { border: 1px solid #3a3a42; border-radius: 4px; }
QTabBar::tab { background: #232329; color: #c9c9d1; padding: 6px 14px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
QTabBar::tab:selected { background: #2e2e36; color: #9ecbff; }
QScrollArea { background-color: transparent; border: none; }
QSplitter::handle { background-color: #2b2b33; }
QSplitter::handle:horizontal { width: 3px; }
"""
