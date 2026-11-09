"""Interface graphique PySide6 (Qt) — macOS et Windows. Thème sombre « moderne » (Fusion).

v0.8.18 :
  - fichier repris TEL QUEL (mainwindow-6.py) : flèches ▼/▶, « Darks » repliée par défaut,
    bouton « Masque ML (ONNX)… » déplacé SOUS la ligne Seuil/Bande/Rés., libellé
    « Affiner la luminance », boîte « Stacking Options », panneau 360 px ;
  - bouton « Reset » à côté de Seuil/Bande/Rés. : remet les paramètres de LUMINANCE
    aux valeurs par défaut (« Affiner la luminance » cochée, Seuil 0 = auto,
    Bande 40 px, Rés. 2000 px) — méthode _reset_lum_params.

v0.8.17 :
  - panneau gauche : AUCUNE barre de scroll horizontale possible — policy AlwaysOff +
    horizontalScrollBar().setVisible(False) + QSS « QScrollBar:horizontal { height: 0px; } »
    sur panel_scroll (catch-all QAbstractScrollArea sur log/listes inclus) ; panel.setMinimumWidth(0),
    combo_lp.setMinimumWidth(0). Seule la barre VERTICALE peut encore apparaître (AsNeeded).

v0.8.16 :
  - PLUS DE BARRE DE SCROLL HORIZONTALE dans le panneau gauche : le log revient à la ligne
    (WidgetWidth) + ScrollBarAlwaysOff, les listes ont ScrollBarAlwaysOff + ElideMiddle,
    panel_scroll reste ScrollBarAlwaysOff (seule la barre verticale peut apparaître) ;
    la ligne Seuil/Bande/Rés. est encore resserrée (marges 0, spins 46/46/62 px) et
    combo_lp_mask.setMinimumWidth(0) pour tenir dans les 340 px sans déborder.

v0.8.15 :
  - box PL : nouveau spin « Fondu » (auto/0-200 px, défaut « auto » = min(h,l)/96, 0 = masque dur)
    = lp_feather exposé en UI (avant : CLI --lp-feather seul) — anti-halo réglable autour de
    l'avant-plan ; transmis au pipeline ET à « Réappliquer la PL » (méthode _lp_feather_value).

v0.8.14 (sur base v0.8.12, tuilage v0.8.13 abandonné) — pack « détail du bord » :
  - « Bande » (0-200 px, défaut 40) : dans une bande de ±N px autour du bord du masque ONNX,
    le bord est RECALÉ sur la luminance pleine résolution (lum <= seuil) — les fines branches
    sombres sur ciel lumineux sont détectées à la résolution de l'image, le ciel loin du bord
    reste intact (pas d'artefacts) ;
  - « Seuil » (0-255, défaut 0 = Otsu auto) : seuil de luminance réglable pour l'affinage ;
  - « Rés. » (800-3200 px, défaut 2000) : résolution de travail du masque ML (1200 avant) ;
  - les 3 réglages tiennent en UNE ligne compacte dans la box Masque (panneau 340 px préservé) ;
  - CLI : --mask-edge-band N (0 = off, défaut 40), --mask-lum-thr existe, --mask-max-side N (défaut 2000).

v0.8.12 :
  - SUPPRESSION de l'élargissement +2 px du masque ML d'empilement (grow 2 -> 0, il
    rognait le ciel au bord des objets) ;
  - affinage luminance ANCRÉ sur le masque ONNX de base : un pixel ne change de classe
    que s'il est RELIÉ (dilatation ~2,5 % du petit côté) au blob ONNX correspondant —
    fini les îlots rouges sans lien en haut de l'image (coins vignettés sombres) ;
  - granularité : lissage médian 5x5 du masque binaire (bords en escalier du sur-
    échantillonage 512 px) + suppression des petites composantes isolées (< 0,15 % de
    l'image) ; CLI : --auto-mask ml applique grow=0 comme l'UI.

v0.8.11 :
  - masque ML d'EMPILEMENT affiné par la LUMINOSITÉ (case « Affiner avec la luminance »,
    cochée par défaut) : après le masque ONNX, les pixels SOMBRES (seuil Otsu auto)
    sont forcés en avant-plan, et les grandes zones TRÈS LUMINEUSES (lueur d'horizon,
    que le modèle ONNX classe « avant-plan ») basculent en ciel — CLI : --no-mask-lum,
    --mask-lum-thr N ;
  - en « ML GraXpert », le combo « Masque PL » est aussi GRISÉ (correction globale, sans masque) ;
  - contraste renforcé pour tous les widgets désactivés (texte #6f6f7a sur fond #222228).

v0.8.10 :
  - méthode « ML GraXpert » = correction sur TOUTE l'image : « Ciel uniquement » est
    automatiquement grisé + décoché (le modèle ONNX protège l'avant-plan, y compris la
    lueur d'horizon classée « avant-plan » par le masque) — c'est ce qui laissait le
    gradient en bas de l'image ; le pipeline force aussi sky_only=False pour GraXpert.

v0.8.9 :
  - « Masque PL = ML ONNX (auto) » : masque ONNX 100 % BRUT — plus AUCUN lissage de proba
    (proba_blur=0), ni morphologie, ni smooth 3x3, ni dilatation, ni dégradé vertical :
    seul le seuil 0,5 de la probabilité « ciel » est utilisé ;
  - le masque PL réellement utilisé est sauvegardé en « <sortie>_maskPL.png » (pipeline,
    réapplication UI et --lp-redo) avec le % de ciel, pour diagnostiquer la lueur d'horizon.

v0.8.8 :
  - boîtes repliables : petite FLÈCHE dans le titre — ▾ ouverte / ▸ fermée — à la place de
    la case à cocher (indicateur masqué en QSS) ;
  - « astrostack_result_avant_PL.tif » n'est écrit qu UNE SEULE FOIS (si le fichier existe
    déjà, il est conservé tel quel) ;
  - fond « synthetic flat » plus « serré » (max_small 768, sigma min/12) pour mieux capturer
    la lueur d'horizon sans manger les étoiles.

v0.8.7 :
  - « Masque PL = ML ONNX (auto) » utilise désormais le masque ONNX **BRUT** pour la
    correction PL : plus de dilatation +2 px (grow=0) ni de dégradé vertical « haut = ciel »
    (top_gradient=False) — ces réglages restent réservés au masque d'EMPILEMENT.

v0.8.6 :
  - ANTI-HALO (fin) : le poids de fusion ciel/avant-plan est dilaté de 1,5*sigma avant
    lissage (au lieu de 0,5*sigma) — le bord du ciel reçoit ~93 % de la correction
    (au lieu de ~70 %), le halo résiduel collé à l'avant-plan doit disparaître ;
  - FORCE de la correction PL réglable jusqu'à 150 % (sur-correction possible du gradient).

v0.8.5 :
  - ANTI-HALO (suite) : le fond « synthetic flat » n'est estimé que sur du ciel « pur »
    (érosion du masque ciel avant le flou normalisé) et le poids de fusion est dilaté de
    sigma/2 avant lissage — la correction s'applique à fond côté ciel jusqu'au bord du
    masque ; le liseré clair autour de la balle doit disparaître ;
  - « Masque PL » dans la box Pollution lumineuse : « Peint (UI) » (défaut) ou
    « ML ONNX (auto) » — régénère le masque ONNX depuis la référence pour la correction PL
    (CLI : --lp-mask {painted,ml}) ; aussi pris en compte par « Réappliquer la PL ».

v0.8.4 :
  - CHECKBOX DES BOÎTES REPLIABLES distinctes des autres cases : pastille ronde 16 px
    (bordure bleue, remplie quand cochée) au lieu du carré 14 px standard (QSS #cbHeader) ;
  - SAUVEGARDE AVANT / APRÈS correction PL : si la PL est activée, le stack linéaire est
    aussi écrit dans « astrostack_result_avant_PL.tif » avant la correction ;
  - BOUTON « Réappliquer la PL sur le dernier stack » (box Pollution lumineuse) : réapplique
    la correction avec les réglages actuels sur le dernier empilement terminé, SANS refaire
    l'alignement ni le stacking (base non corrigée conservée en mémoire via meta_out) ;
    CLI : --lp-redo resultat.tif écrit <base>_pl.tif ;
  - fond « synthetic flat » estimé à 512 px (au lieu de 256) : halo résiduel encore réduit.

v0.8.3 :
  - raccourcis Ctrl+D / Ctrl+Maj+D CORRIGÉS : pointent enfin vers les DARKS
    (self._darks / list_darks, et non plus vers la liste des lights) ;
  - masque ONNX (ML) : avant-plan dilaté de 2 px partout (cv2.dilate, iterations=2) ;
  - correction PL : le fond « synthetic flat » est estimé UNIQUEMENT sur les pixels du
    ciel (flou normalisé masqué) — l'avant-plan lumineux (ex. balle de foin) ne contamine
    plus le fond, ce qui supprimait le halo résiduel ; lissage du bord du masque renforcé
    (sigma auto = min(h,l)/96 au lieu de /128).

v0.8.2 :
  - fichier utilisateur repris tel quel (mainwindow-4.py : panneau 340 px, libellé
    « Ciel uniquement ») ;
  - PANNEAU GAUCHE PLUS HAUT (fenêtre 1120x860 par défaut, min 880x620) et la fenêtre
    S'AGRANDIT EN HAUTEUR quand on déplie une boîte (et se réduit au repli) : les boîtes
    ne sont plus compressées (widgetResizable(True) + hauteurs mini listes/log) ; si ça
    ne tient pas à l'écran, la barre de défilement verticale du panneau prend le relais ;
  - correction PL : le masque de fusion ciel/avant-plan est LISSÉ (flou gaussien, sigma
    auto) — v0.8.2 dans lpcorrect.py, pour supprimer les halos au bord du masque peint.

v0.8.0 :
  - CORRECTION DE LA POLLUTION LUMINEUSE (post-empilement, linéaire, avant export TIFF) :
    nouvelle box repliée « Pollution lumineuse » (case + méthode + force % + « ciel
    uniquement ») ; méthodes : « Synthetic flat » (défaut, fond lissé estimé sur le ciel)
    et « ML GraXpert (ONNX) » (modèle officiel background-extraction 1.0.1, ~218 Mo,
    licence CC BY-NC-SA non commercial, téléchargé dans ~/.astrostack/) ; CLI :
    --lp-correct / --lp-method / --lp-strength / --no-lp-sky-only.
v0.7.15 :
  - fichier utilisateur repris tel quel (mainwindow-3.py : raccourcis Ctrl+D / Ctrl+Maj+D
    et gomme déplacée sur X) ;
  - TAILLE DU PINCEAU réglable au clavier et à la souris : touches + / - (et = / _ pour
    les claviers US) par pas de 10 px, et MOLETTE sur l'aperçu de la référence
    (eventFilter, +/- 10 px) ; toujours synchronisé avec le slider « Pinceau ».
v0.7.14 :
  - correction utilisateur reprise telle quelle (mainwindow-2.py) : panneau latéral FIXE
    à 320 px (min=max=320, widgetResizable False), « Options » déplacé SOUS « Masque
    d'avant-plan », libellé « Gradient Ciel favorisé », bouton « ▶ Lancer » ;
  - RACCOURCIS CLAVIER (QShortcut ; Ctrl = ⌘ sur macOS) : Ctrl+O ajouter des images,
    Ctrl+Maj+O ajouter un dossier, Ctrl+M masque ML, Ctrl+Entrée lancer, Échap arrêter,
    G gomme on/off, I inverser le masque, F1 aide.
v0.7.12 :
  - panneau latéral : retour à la largeur d'avant (min 208 / max 288, splitter ~256 px) —
    un min à 320 forçait le panneau à ~288–320 px, comprimait l'aperçu à droite (barre de
    défilement qui réapparaissait) ; contenu compacté (listes 60 px, log 80 px, espacements
    réduits) et fenêtre par défaut 1120x760 pour éviter la barre verticale du panneau.
v0.7.11 :
  - « Options » devient une CollapsibleBox repliable comme « Darks/Flats/Bias » ;
  - dégradé ONNX « haut = ciel » : FACTEUR RÉGLABLE (spin 0–3, défaut 0,5) à côté de la
    case à cocher, passé au masque via top_boost (0 = neutre, 1,0+ = fort).
v0.7.10 :
  - CORRECTION « Ajouter… » : le sélecteur embarqué (QFileDialog non natif dans une QDialog)
    pouvait s'ouvrir SANS liste de fichiers. Retour aux boîtes de dialogue NATIVES
    (QFileDialog.getOpenFileNames / getExistingDirectory) pour « Ajouter… », « Dossier… »
    et le dossier de sortie : navigation garantie ; sous macOS, le panneau natif affiche
    déjà les vignettes des images comme le Finder en mode icônes. Dernier dossier
    utilisé mémorisé (QSettings « last_dir ») ;
  - bouton « Lancer » : texte seul (sans emoji ni « empilement »).
v0.7.9 :
  - RÉFÉRENCE changeable en CLIQUANT sur une image de la liste (mise en gras) ; label
    simplifié « Référence : nom du fichier » ; défaut = image du milieu ;
  - bouton « ⏹ Arrêter » : annulation coopérative de l'empilement (should_cancel dans
    le pipeline, CancelledError) ;
  - réglage « classe ciel » retiré de l'UI (reste ASTROSTACK_SKY_CLASS en env) ;
  - masque ONNX : dégradé vertical « haut = ciel » sur la proba, avec case à cocher
    pour activer/désactiver ;
  - sélecteur d'images : dialogue dédié avec volet d'aperçu À DROITE (style Finder),
    image contenue avec barre de déroulement, vignette RAW embarquée.
v0.7.8 :
  - bouton « Auto » et son seuil (0–255) SUPPRIMÉS de l'UI (inutiles) — le CLI garde --auto-mask ;
  - classe « ciel » du modèle ONNX réglable DANS l'UI (spin 0–149, défaut 2 = ADE20K sky,
    pré-rempli depuis ASTROSTACK_SKY_CLASS si défini) et passée au masque ML, loguée.
v0.7.7 :
  - masque ML : inférence sur la RÉFÉRENCE à pleine résolution utile (plafond 1200 px,
    comme en v0.6.0), et non plus sur l'aperçu réduit à la fenêtre — c'est ce qui
    donnait « ~100 % de ciel » en v0.7.5/0.7.6. Le masque est ensuite redimensionné à
    l'aperçu. Repli luminance + logs de proba conservés.
v0.7.6 :
  - masque ONNX : plus de « toute l'image en avant-plan » sur ciel nocturne — repli
    luminance si le modèle ne reconnaît aucun ciel (proba < 0.02), correction du layout
    des logits (canaux dernier), diagnostics logués (proba ciel moyenne/max) ;
  - label « Référence : … » dans la boîte Masque : l'image de base (milieu de liste)
    est affichée dans l'UI ;
  - tooltips explicites pour « Auto » (seuil de luminance) et le chiffre à côté (0–255).
v0.7.5 :
  - CORRECTION crash aperçu sélecteur : currentChanged émet une str en PySide6/Qt6
    (plus une QUrl) — géré des deux côtés ;
  - case « Gomme » : état coché ENFIN VISIBLE en QSS sombre (indicateur bleu) + log
    « Gomme activée / Pinceau » ;
  - l'aperçu de la RÉFÉRENCE tient dans la fenêtre : réduit pour entrer dans la zone
    visible (plafond 1200 px), barre de déroulement si la fenêtre est plus petite.
v0.7.4 :
  - CORRECTION crash « Ajouter… » : QFileDialog.setPreviewWidget n'existe plus en Qt6 /
    PySide6 (AttributeError). Le volet d'aperçu est maintenant ajouté MANUELLEMENT dans
    le layout (grille) du dialogue non natif — rangée du bas, pleine largeur, hauteur fixe,
    image contenue avec barre de déroulement — avec repli sur l'API officielle si elle
    existe, et try/except : l'aperçu ne peut plus jamais empêcher la sélection de fichiers.
v0.7.3 :
  - espace alloué du panneau élargi (208–288 px, splitter 256/864) : les boîtes tiennent
    sans être coupées, défilement horizontal toujours off ;
  - sélecteur de fichiers : panneau d'APERÇU des images (comme avant), contenu DANS la
    boîte de dialogue (volet 212x180) avec barre de déroulement si l'image dépasse,
    vignette JPEG embarquée pour les RAW via rawpy ;
  - l'aperçu du RÉSULTAT est ajusté à la taille de la fenêtre (la barre de déroulement
    intervient si la fenêtre est plus petite).
v0.7.2 :
  - panneau 172–236 px avec contenu compacté pour TENIR dans la largeur allouée
    (pas de défilement horizontal : listes 72 px, textes raccourcis, cases pleine largeur) ;
  - boîtes repliables : retour au look d'avant « cadre + case à cocher », mais avec une
    QCheckBox d'en-tête réellement cliquable (le QGroupBox checkable + QSS ne répondait plus) ;
  - boîtes de dialogue de fichiers dimensionnées (720x460, non natives) : elles tiennent
    dans l'espace alloué.
v0.7.1 :
  - panneau de gauche encore réduit (168–248 px, splitter 216/904, listes 84 px) ;
  - sections repliables réparées : l'en-tête est un QToolButton cliquable (flèche ▾/▸),
    le QGroupBox checkable + QSS ne répondait plus au clic sur Flats/Bias.
v0.7.0 :
  - apparence sombre (QSS) + style Fusion ;
  - panneau de gauche plus étroit (230–320 px) ;
  - sections « Images / Darks / Flats / Bias » repliables individuellement (case à cocher,
    repliées sur la hauteur) — par défaut seules Images et Darks sont ouvertes ;
  - ajout de fichiers SANS réouverture de boîte de dialogue : deux boutons par liste
    (« Ajouter… » = fichiers, « Dossier… » = dossier entier) ;
  - chaque image a une croix ✕ à côté de son nom pour la supprimer de la liste ;
  - l'aperçu de la RÉFÉRENCE est chargé automatiquement dès que des lights sont ajoutées ;
  - la référence est par défaut l'image DU MILIEU de la liste (ref_index = N//2).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PySide6.QtCore import QEvent, Qt, QSettings, QThread, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .maskwidget import MaskEditor
from .pipeline import run_pipeline
from .preview import u16_to_qimage
from .rawio import SUPPORTED_EXTS, list_image_files, load_image_linear

IMG_FILTER = "Images (*" + " *".join(SUPPORTED_EXTS) + ")"
SETTINGS = QSettings("AstroStack", "AstroStack")

DARK_QSS = """
QMainWindow, QWidget { background-color: #1b1b1f; color: #e8e8ec; }
QGroupBox { border: 1px solid #3a3a42; border-radius: 6px; margin-top: 12px; padding: 4px 6px 6px 6px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 4px; color: #9ecbff; }
QGroupBox::indicator { width: 14px; height: 14px; border: 1px solid #45454f; border-radius: 3px; background-color: #2e2e36; }
QGroupBox::indicator:checked { background-color: #4a90d9; border-color: #6db3f2; }
QGroupBox#CollapsibleBox { margin-top: 4px; padding: 2px 6px 6px 6px; background-color: #202027; }
QCheckBox#cbHeader { font-weight: 600; color: #9ecbff; font-size: 12px; spacing: 4px; }
QCheckBox#cbHeader::indicator { width: 0px; height: 0px; border: none; background: transparent; }  /* v0.8.8 : indicateur masqué, flèche ▸/▾ dans le titre */
QListWidget { background-color: #232329; border: 1px solid #3a3a42; border-radius: 4px; color: #e8e8ec; }
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


class StackWorker(QThread):
    progressed = Signal(int, int, int, str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._cancel = False

    def cancel(self) -> None:
        """Demande l'annulation coopérative (vérifiée à chaque étape du pipeline)."""
        self._cancel = True

    def run(self) -> None:
        from .pipeline import CancelledError

        try:
            result = run_pipeline(
                progress_cb=lambda s, i, n, m: self.progressed.emit(s, i, n, m),
                should_cancel=lambda: self._cancel,
                **self._cfg,
            )
            self.done.emit(result)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__} : {exc}")



class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AstroStack — empilement de paysages de nuit")
        self.setMinimumSize(880, 620)
        self.resize(1120, 860)   # v0.8.2 : panneau gauche plus haut par défaut

        self._lights: list[str] = []
        self._ref_idx: int | None = None  # None = image du milieu par défaut ; set par clic dans la liste
        self._darks: list[str] = []
        self._flats: list[str] = []
        self._biases: list[str] = []
        self._out_dir = os.getcwd()
        self._last_dir = str(SETTINGS.value("last_dir", "")) or os.path.expanduser("~")
        self._worker: StackWorker | None = None
        self._result_full: QPixmap | None = None  # aperçu résultat, pleine résolution
        self._stack_base = None   # v0.8.4 : stack linéaire AVANT correction PL (pour réappliquer)
        self._meta: dict = {}     # v0.8.4 : meta_out rempli par le pipeline (sky, fg_w, stack_raw…)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self.splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.splitter)

        # ---------- panneau de gauche (étroit, scrollable, repliable) ----------
        left = QVBoxLayout()
        left.setContentsMargins(4, 4, 4, 4)
        left.setSpacing(3)

        box_img = CollapsibleBox("Images", checked=True)   # ouvertes par défaut
        box_dark = CollapsibleBox("Darks (optionnel)", checked=False)   # ouvertes par défaut
        box_flat = CollapsibleBox("Flats (optionnel)", checked=False)  # repliées
        box_bias = CollapsibleBox("Bias (optionnel)", checked=False)   # repliées
        self.list_lights = self._make_list_row(box_img, self._lights)
        self.list_lights.setCursor(Qt.PointingHandCursor)
        self.list_lights.setToolTip("Cliquez sur une image pour en faire la RÉFÉRENCE (image de base de l'alignement).")
        self.list_lights.itemClicked.connect(self._on_light_clicked)
        self.list_darks = self._make_list_row(box_dark, self._darks)
        self.list_flats = self._make_list_row(box_flat, self._flats)
        self.list_bias = self._make_list_row(box_bias, self._biases)
        for b in (box_img, box_dark, box_flat, box_bias):
            left.addWidget(b)

        box_out = QGroupBox("Dossier de sortie")
        lay_out = QHBoxLayout(box_out)
        self.edit_out = QLabel(self._out_dir)
        self.edit_out.setWordWrap(True)
        btn_browse = QPushButton("Parcourir…")
        btn_browse.clicked.connect(self._browse_out)
        lay_out.addWidget(self.edit_out, 1)
        lay_out.addWidget(btn_browse)
        left.addWidget(box_out)


        box_mask = QGroupBox("Masque d'avant-plan")
        lay_mask = QVBoxLayout(box_mask)
        self.lbl_ref = QLabel("Référence : — (ajoutez des images)")
        self.lbl_ref.setWordWrap(True)
        self.lbl_ref.setStyleSheet("color: #9ecbff; font-weight: 600;")
        self.lbl_ref.setToolTip("Image de base utilisée pour l'alignement et l'aperçu : c'est l'image du MILIEU de la liste (modifiable en CLI avec --ref-index N).")
        lay_mask.addWidget(self.lbl_ref)
        btn_mask = QPushButton("Recharger l'aperçu…")
        btn_mask.setToolTip("Recharge l'aperçu de la référence (image du milieu de la liste des Images)")
        btn_mask.clicked.connect(lambda: self._load_reference_preview(force=True))
        lay_mask.addWidget(btn_mask)
        row_brush = QHBoxLayout()
        row_brush.addWidget(QLabel("Pinceau"))
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(4, 300)
        self.slider_brush.setValue(40)
        self.slider_brush.valueChanged.connect(lambda v: setattr(self.mask_editor, "brush_size", v))
        row_brush.addWidget(self.slider_brush, 1)
        lay_mask.addLayout(row_brush)
        row_tools = QHBoxLayout()
        self.chk_erase = QCheckBox("Gomme")
        self.chk_erase.toggled.connect(self._on_erase_toggled)
        row_tools.addWidget(self.chk_erase)
        row_tools.addStretch(1)
        lay_mask.addLayout(row_tools)
        #row_tools2 = QHBoxLayout()
        btn_inv = QPushButton("Inverser")
        btn_inv.clicked.connect(lambda: self.mask_editor.invert_mask())
        row_tools.addWidget(btn_inv)
        btn_clear = QPushButton("Effacer")
        btn_clear.clicked.connect(lambda: self.mask_editor.clear_mask())
        row_tools.addWidget(btn_clear)


        row_sky = QHBoxLayout()
        self.chk_top_sky = QCheckBox("Gradient Ciel favorisé")
        self.chk_top_sky.setChecked(True)
        self.chk_top_sky.setToolTip("Masque ML : applique un dégradé vertical à la probabilité « ciel » — le HAUT de l'image est favorisé comme ciel (typique en astrophoto). Décochez pour désactiver.")
        row_sky.addWidget(self.chk_top_sky, 1)
        row_sky.addWidget(QLabel("Facteur"))
        self.spin_top_boost = QDoubleSpinBox()
        self.spin_top_boost.setRange(0.0, 3.0)
        self.spin_top_boost.setSingleStep(0.1)
        self.spin_top_boost.setDecimals(2)
        self.spin_top_boost.setValue(0.5)
        self.spin_top_boost.setMaximumWidth(80)
        self.spin_top_boost.setToolTip("Force du dégradé « haut = ciel » : 0 = neutre, 0,5 = normal (défaut), 1,0 et + = de plus en plus fort. N'agit que si la case est cochée.")
        row_sky.addWidget(self.spin_top_boost)
        lay_mask.addLayout(row_sky)
        self.chk_lum_refine = QCheckBox("Affiner la luminance")
        self.chk_lum_refine.setChecked(True)
        self.chk_lum_refine.setToolTip(
            "v0.8.11 — affine le masque ML (empilement) avec la luminosité de la référence : "
            "pixels SOMBRES (seuil Otsu automatique) forcés en avant-plan ; grandes zones TRÈS "
            "lumineuses (lueur d'horizon, que le modèle ONNX classe souvent « avant-plan ») basculées "
            "en ciel. Décochez pour garder le masque ONNX seul."
        )
        lay_mask.addWidget(self.chk_lum_refine)
        row_lum = QHBoxLayout()  # v0.8.14 : UNE seule ligne compacte (3 petits spins) pour ne pas faire grossir le panneau
        row_lum.setContentsMargins(0, 0, 0, 0)  # v0.8.16 : pas de marge -> tient dans 340 px sans barre horizontale
        row_lum.setSpacing(2)
        row_lum.addWidget(QLabel("Seuil"))
        self.spin_lum_thr = QSpinBox()
        self.spin_lum_thr.setRange(0, 255)
        self.spin_lum_thr.setValue(0)
        self.spin_lum_thr.setMaximumWidth(46)
        self.spin_lum_thr.setToolTip(
            "v0.8.14 — seuil de luminance 0..255 pour l'affinage du masque ML (0 = auto, seuil d'Otsu). "
            "Plus haut = plus de pixels sombres classés avant-plan."
        )
        row_lum.addWidget(self.spin_lum_thr)
        row_lum.addWidget(QLabel("Bande"))
        self.spin_edge_band = QSpinBox()
        self.spin_edge_band.setRange(0, 200)
        self.spin_edge_band.setValue(40)
        self.spin_edge_band.setMaximumWidth(46)
        self.spin_edge_band.setToolTip(
            "v0.8.14 — largeur en px de la bande autour du bord du masque ONNX où le bord est RECALÉ "
            "sur la luminance pleine résolution (détecte les fines branches sombres sur ciel lumineux). "
            "0 = désactivé."
        )
        row_lum.addWidget(self.spin_edge_band)
        row_lum.addWidget(QLabel("Rés."))
        self.spin_mask_res = QSpinBox()
        self.spin_mask_res.setRange(800, 3200)
        self.spin_mask_res.setSingleStep(400)
        self.spin_mask_res.setValue(2000)
        self.spin_mask_res.setMaximumWidth(62)
        self.spin_mask_res.setSuffix(" px")
        self.spin_mask_res.setToolTip(
            "v0.8.14 — grand côté (px) de l'image de travail pour le calcul du masque ML : "
            "plus grand = détails plus fins, mais plus lent. 2000 px recommandé (1200 avant)."
        )
        row_lum.addWidget(self.spin_mask_res)
        self.btn_lum_reset = QPushButton("Reset")   # v0.8.18 : remet les paramètres de luminance par défaut
        self.btn_lum_reset.setMaximumWidth(52)
        self.btn_lum_reset.setToolTip(
            "v0.8.18 — remet les paramètres de luminance aux valeurs par défaut : "
            "« Affiner la luminance » cochée, Seuil 0 (auto), Bande 40 px, Rés. 2000 px."
        )
        self.btn_lum_reset.clicked.connect(self._reset_lum_params)
        row_lum.addWidget(self.btn_lum_reset)
        lay_mask.addLayout(row_lum)
        btn_ml = QPushButton("Masque ML (ONNX)…")
        btn_ml.setToolTip("Segmente le ciel avec un modèle ONNX (segformer ADE20K), téléchargeable automatiquement. "
                         "Installez onnxruntime : pip install onnxruntime")
        btn_ml.clicked.connect(self._ml_mask)
        lay_mask.addWidget(btn_ml)
        left.addWidget(box_mask)


        box_opt = CollapsibleBox("Stacking Options", checked=True)   # repliable comme Darks/Flats/Bias
        form = QFormLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.setMinimumWidth(0)
        self.combo_mode.addItems(["Sigma clipping", "Moyenne simple"])
        form.addRow("Empilement", self.combo_mode)
        self.spin_kappa = QDoubleSpinBox()
        self.spin_kappa.setRange(1.0, 10.0)
        self.spin_kappa.setValue(2.5)
        self.spin_kappa.setMaximumWidth(90)
        form.addRow("Kappa (sigma)", self.spin_kappa)
        self.combo_scale = QComboBox()
        self.combo_scale.setMinimumWidth(0)
        self.combo_scale.addItems(["100 %", "50 %", "25 %"])
        self.combo_scale.setToolTip("100 % = pleine résolution de travail")
        form.addRow("Résolution", self.combo_scale)
        self.chk_ecc = QCheckBox("Affiner l'alignement (ECC)")
        self.chk_ecc.setChecked(True)
        self.chk_ecc.setToolTip(
            "Après l'alignement par étoiles (homographie), un affinage photométrique ECC "
            "réajuste chaque pose sur le ciel : corrige les traînées résiduelles, surtout "
            "sur les bords de l'image. Désactivez uniquement si une pose résiste mal."
        )
        form.addRow(self.chk_ecc)
        self.chk_local = QCheckBox("Warp local (TPS) — coins")
        self.chk_local.setChecked(True)
        self.chk_local.setToolTip(
            "Warp local par thin-plate spline à partir des étoiles appariées, réparties sur "
            "tout le champ (coins compris) : corrige ce qu'une homographie ne peut pas "
            "modéliser, notamment la distorsion de l'objectif — cause classique des traînées "
            "d'étoiles dans les coins. Désactivez si une pose pose problème."
        )
        form.addRow(self.chk_local)
        box_opt.addLayout(form)
        left.addWidget(box_opt)

        box_lp = CollapsibleBox("Pollution lumineuse", checked=False)   # repliée par défaut
        lay_lp = QVBoxLayout()
        lay_lp.setContentsMargins(2, 2, 2, 2)
        lay_lp.setSpacing(2)
        self.chk_lp = QCheckBox("Corriger la pollution lumineuse")
        self.chk_lp.setToolTip("Post-empilement, en linéaire : soustrait le dégradé de fond de ciel (additif).")
        lay_lp.addWidget(self.chk_lp)
        row_lp1 = QHBoxLayout()
        row_lp1.addWidget(QLabel("Méthode"))
        self.combo_lp = QComboBox()
        self.combo_lp.setMinimumWidth(0)  # v0.8.17 : peut rétrécir -> tient dans 340 px
        self.combo_lp.addItems(["Synthetic flat (défaut)", "ML GraXpert (ONNX)"])
        self.combo_lp.setToolTip(
            "Synthetic flat : fond lissé estimé sur le ciel (rapide, sans modèle). "
            "ML GraXpert : modèle ONNX ~218 Mo (licence CC BY-NC-SA, usage non commercial), "
            "téléchargé au premier usage dans ~/.astrostack/."
        )
        row_lp1.addWidget(self.combo_lp, 1)
        lay_lp.addLayout(row_lp1)
        row_lp2 = QHBoxLayout()
        row_lp2.addWidget(QLabel("Force"))
        self.spin_lp = QSpinBox()
        self.spin_lp.setRange(0, 150)   # v0.8.6 : jusqu'à 150 % (sur-correction du gradient résiduel)
        self.spin_lp.setValue(100)
        self.spin_lp.setSuffix(" %")
        self.spin_lp.setMaximumWidth(80)
        self.spin_lp.setToolTip("Force de la correction PL : 100 % = correction complète, "
                                "> 100 % = sur-correction (peut aider à tuer un halo résiduel), 0 % = off.")
        row_lp2.addWidget(self.spin_lp)
        self.chk_lp_sky = QCheckBox("Ciel uniquement")
        self.chk_lp_sky.setChecked(True)
        self.chk_lp_sky.setToolTip("N'applique la correction que sur le ciel (masque avant-plan) : l'avant-plan reste intact. "
                                  "La transition au bord du masque est lissée (anti-halos, v0.8.2). "
                                  "Décochez pour corriger toute l'image (aucun artefact de masque).")
        row_lp2.addWidget(self.chk_lp_sky)
        row_lp2.addStretch(1)
        lay_lp.addLayout(row_lp2)
        row_lp3 = QHBoxLayout()
        row_lp3.addWidget(QLabel("Masque PL"))
        self.combo_lp_mask = QComboBox()
        self.combo_lp_mask.setMinimumWidth(0)  # v0.8.16 : peut rétrécir -> pas de barre horizontale dans le panneau 340 px
        self.combo_lp_mask.addItems(["Peint (UI)", "ML ONNX (auto)"])
        self.combo_lp_mask.setToolTip(
            "Masque utilisé POUR la correction de la pollution lumineuse (v0.8.5) : "
            "« Peint (UI) » = le masque peint dans l'éditeur ; « ML ONNX (auto) » = masque ONNX "
            "régénéré depuis la référence, utilisé BRUT (v0.8.7 : sans dilatation +2 px ni dégradé "
            "vertical — réservés au masque d'empilement ; nécessite onnxruntime)."
        )
        row_lp3.addWidget(self.combo_lp_mask, 1)
        row_lp3.addWidget(QLabel("Fondu"))
        self.spin_lp_feather = QDoubleSpinBox()
        self.spin_lp_feather.setRange(-1.0, 200.0)
        self.spin_lp_feather.setSingleStep(2.0)
        self.spin_lp_feather.setDecimals(0)
        self.spin_lp_feather.setValue(-1.0)
        self.spin_lp_feather.setSpecialValueText("auto")
        self.spin_lp_feather.setMaximumWidth(70)
        self.spin_lp_feather.setSuffix(" px")
        self.spin_lp_feather.setToolTip(
            "v0.8.15 — sigma (px) du lissage de la transition au bord du masque en correction PL "
            "(anti-halo autour de l'avant-plan, ex. la balle). « auto » (défaut) = min(h,l)/96 ; "
            "0 = masque dur. CLI : --lp-feather."
        )
        row_lp3.addWidget(self.spin_lp_feather)
        lay_lp.addLayout(row_lp3)
        self.combo_lp.currentIndexChanged.connect(self._on_lp_method_changed)
        self._on_lp_method_changed(self.combo_lp.currentIndex())
        self.btn_lp_redo = QPushButton("Réappliquer la PL sur le dernier stack")
        self.btn_lp_redo.setEnabled(False)
        self.btn_lp_redo.setToolTip(
            "v0.8.4 : réapplique la correction de la pollution lumineuse avec les réglages "
            "actuels (méthode / force / ciel uniquement) sur le DERNIER empilement terminé, "
            "sans refaire l'alignement ni l'empilement — sauve un nouveau TIFF. "
            "La base non corrigée est conservée : on peut donc changer les réglages et réappliquer."
        )
        self.btn_lp_redo.clicked.connect(self._redo_lp)
        lay_lp.addWidget(self.btn_lp_redo)
        box_lp.addLayout(lay_lp)
        left.addWidget(box_lp)


        row_run = QHBoxLayout()
        self.btn_run = QPushButton("▶ Lancer")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px; background-color: #2f5d8a; border-color: #4a90d9;")
        self.btn_run.clicked.connect(self._run)
        row_run.addWidget(self.btn_run, 1)
        self.btn_stop = QPushButton("⏹ Arrêter")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Annuler l'empilement en cours")
        self.btn_stop.clicked.connect(self._on_stop)
        row_run.addWidget(self.btn_stop)
        left.addLayout(row_run)

        self.progress = QProgressBar()
        left.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(56)   # v0.8.2 : anti-compression
        self.log.setMaximumHeight(80)
        self.log.setLineWrapMode(QPlainTextEdit.WidgetWidth)          # v0.8.16 : le log revient à la ligne -> aucune barre horizontale
        self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # v0.8.16 : aucune barre de scroll horizontale en bas du panneau
        left.addWidget(self.log, 1)

        panel = QWidget()
        panel.setLayout(left)
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setWidgetResizable(True)   # v0.8.2 : le contenu ne se compresse plus (défilement si trop haut)
        self.panel_scroll.setFrameShape(QFrame.NoFrame)
        self.panel_scroll.setWidget(panel)
        self.panel_scroll.setMinimumWidth(360)   # panneau étroit (splitter ~256 px), comme avant
        self.panel_scroll.setMaximumWidth(360)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # le contenu tient dans la largeur allouée (v0.8.16 : aucun horizontal bar, même si un widget déborde)
        self.panel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)     # v0.8.16 : seule la barre VERTICALE reste possible (défilement haut/bas)
        # v0.8.17 : AUCUNE barre HORIZONTALE visible dans le panneau gauche, quoi qu'il arrive :
        # policy off + scrollbar elle-même cachée + QSS hauteur 0 sur toute barre horizontale
        # (y compris celles des widgets internes : log, listes…) — catch-all QAbstractScrollArea
        self.panel_scroll.horizontalScrollBar().setVisible(False)
        self.panel_scroll.widget().setMinimumWidth(0)
        for _w in self.panel_scroll.findChildren(QAbstractScrollArea):
            _w.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            _w.horizontalScrollBar().setVisible(False)
        self.panel_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:horizontal { height: 0px; }"  # même "shown", une barre horizontale fait 0 px de haut -> invisible
        )
        # v0.8.2 : déplier/replier une box ajuste la hauteur de la fenêtre (anti-compression)
        for b in (box_img, box_dark, box_flat, box_bias, box_opt, box_lp):
            b.chk.toggled.connect(self._on_box_toggled)

        # ---------- zone droite ----------
        right = QVBoxLayout()
        topbar = QHBoxLayout()
        self.btn_toggle = QPushButton("<<")
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.setToolTip("Afficher / masquer le panneau de contrôle")
        self.btn_toggle.toggled.connect(self._toggle_panel)
        topbar.addWidget(self.btn_toggle)
        topbar.addStretch(1)
        right.addLayout(topbar)

        self.tabs = QTabWidget()
        self.mask_editor = MaskEditor()
        self.mask_scroll = QScrollArea()
        self.mask_scroll.setWidget(self.mask_editor)
        self.mask_scroll.setAlignment(Qt.AlignCenter)
        self.mask_editor.installEventFilter(self)          # molette souris = taille du pinceau
        self.mask_scroll.viewport().installEventFilter(self)
        self.tabs.addTab(self.mask_scroll, "Référence")
        self.result_label = QLabel("Le résultat apparaîtra ici.")
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidget(self.result_label)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.result_scroll, "Résultat")
        right.addWidget(self.tabs, 1)

        rightw = QWidget()
        rightw.setLayout(right)

        self.splitter.addWidget(self.panel_scroll)
        self.splitter.addWidget(rightw)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setSizes([256, 864])

        geo = SETTINGS.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        if SETTINGS.value("panel_visible", True, type=bool) is False:
            self.btn_toggle.setChecked(False)

        # ---------- raccourcis clavier (Ctrl = ⌘ sur macOS) ----------
        self._shortcuts = {
            "Ctrl+O": ("Ajouter des images…", lambda: self._add_files(self._lights, self.list_lights)),
            "Ctrl+Shift+O": ("Ajouter un dossier d'images…", lambda: self._add_dir(self._lights, self.list_lights)),
            "Ctrl+D": ("Ajouter des darks…", lambda: self._add_files(self._darks, self.list_darks)),
            "Ctrl+Shift+D": ("Ajouter un dossier de darks…", lambda: self._add_dir(self._darks, self.list_darks)),
            "Ctrl+M": ("Masque ML (ONNX)", self._ml_mask),
            "Ctrl+Return": ("Lancer l'empilement", self._run),
            "Escape": ("Arrêter l'empilement", self._on_stop),
            "X": ("Gomme (on/off)", lambda: self.chk_erase.setChecked(not self.chk_erase.isChecked())),
            "I": ("Inverser le masque", lambda: self.mask_editor.invert_mask()),
            "+": ("Pinceau plus gros", lambda: self._bump_brush(+10)),
            "=": ("Pinceau plus gros (clavier US)", lambda: self._bump_brush(+10)),
            "-": ("Pinceau plus petit", lambda: self._bump_brush(-10)),
            "_": ("Pinceau plus petit (clavier US)", lambda: self._bump_brush(-10)),
        }
        for keys, (_tip, fn) in self._shortcuts.items():
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(fn)
        sc_help = QShortcut(QKeySequence("F1"), self)
        sc_help.setContext(Qt.WindowShortcut)
        sc_help.activated.connect(self._show_shortcuts)

        self._log("Prêt. Ajoutez des lights (RAW CR2/DNG/…) — l'aperçu de la référence (image du milieu) se charge automatiquement.")

    # ---------- persistance ----------
    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if getattr(self, "result_scroll", None) is not None:
            self._fit_result()  # l'aperçu du résultat suit la taille de la fenêtre

    def closeEvent(self, ev) -> None:
        SETTINGS.setValue("geometry", self.saveGeometry())
        SETTINGS.setValue("panel_visible", self.btn_toggle.isChecked())
        super().closeEvent(ev)

    def _toggle_panel(self, checked: bool) -> None:
        self.panel_scroll.setVisible(checked)
        self.btn_toggle.setText("<<" if checked else ">>")

    def _on_box_toggled(self, _checked: bool) -> None:
        """v0.8.2 — Déplier / replier une box ajuste la HAUTEUR de la fenêtre pour que le
        panneau gauche affiche tout son contenu (les boîtes ne sont plus compressées) ;
        si ça ne tient pas à l'écran, la barre de défilement verticale prend le relais."""
        panel = self.panel_scroll.widget()
        if panel is None:
            return
        QApplication.processEvents()
        need = panel.sizeHint().height() + 24   # marges du layout + cadre
        screen = QApplication.primaryScreen().availableGeometry()
        target = max(self.minimumHeight(), min(need, screen.height() - 40))
        if abs(target - self.height()) > 4:
            self.resize(self.width(), target)

    # ---------- listes de fichiers (✕ par item, ajout fichiers/dossier séparé) ----------
    def _make_list_row(self, box: CollapsibleBox, backing: list[str]) -> QListWidget:
        lst = QListWidget()
        lst.setMinimumHeight(48)   # v0.8.2 : anti-compression
        lst.setMaximumHeight(60)
        lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # v0.8.16 : pas de barre horizontale dans le panneau
        lst.setTextElideMode(Qt.ElideMiddle)                      # v0.8.16 : noms de fichiers longs tronqués avec « … »
        box.addWidget(lst)
        row = QHBoxLayout()
        btn_files = QPushButton("Ajouter…")   # boîte de dialogue FICHiers (ne se rouvre pas ensuite)
        btn_dir = QPushButton("Dossier…")     # boîte de dialogue DOSSIER, indépendante
        btn_files.clicked.connect(lambda: self._add_files(backing, lst))
        btn_dir.clicked.connect(lambda: self._add_dir(backing, lst))
        row.addWidget(btn_files)
        row.addWidget(btn_dir)
        box.addLayout(row)
        return lst

    def _add_item(self, lst: QListWidget, backing: list[str], path: str) -> None:
        item = QListWidgetItem()
        item.setData(Qt.UserRole, path)
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(4)
        btn = QToolButton()
        btn.setText("✕")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip("Supprimer cette image de la liste")
        btn.clicked.connect(lambda _checked=False, it=item: self._remove_item(lst, backing, it))
        lab = QLabel(os.path.basename(path))
        h.addWidget(btn)
        h.addWidget(lab, 1)
        item.setSizeHint(w.sizeHint())
        lst.addItem(item)
        lst.setItemWidget(item, w)
        backing.append(path)

    def _remove_item(self, lst: QListWidget, backing: list[str], item: QListWidgetItem) -> None:
        row = lst.row(item)
        lst.takeItem(row)
        if 0 <= row < len(backing):
            backing.pop(row)
        self._on_lists_changed()

    def _add_files(self, backing: list[str], lst: QListWidget) -> None:
        # Dialogue NATIF (navigation garantie) : sous macOS, le panneau affiche les
        # vignettes des images comme le Finder en mode icônes / colonnes.
        files, _sel = QFileDialog.getOpenFileNames(
            self,
            "Choisir des images",
            self._last_dir,
            IMG_FILTER + ";;Tous les fichiers (*)",
        )
        if not files:
            return
        self._last_dir = os.path.dirname(files[0]) or self._last_dir
        SETTINGS.setValue("last_dir", self._last_dir)
        for f in files:
            self._add_item(lst, backing, f)
        self._log(f"{len(files)} fichier(s) ajouté(s) ({len(backing)} au total).")
        self._on_lists_changed()

    def _add_dir(self, backing: list[str], lst: QListWidget) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choisir un dossier d'images", self._last_dir)
        if not d:
            return
        files = list_image_files(d)
        for f in files:
            self._add_item(lst, backing, f)
        self._log(f"{len(files)} fichier(s) du dossier {os.path.basename(d)} ({len(backing)} au total).")
        self._on_lists_changed()

    def _on_lists_changed(self) -> None:
        """Dès que la liste des lights change : ajuste la référence et recharge l'aperçu
        si l'utilisateur n'a pas encore peint de masque."""
        if self._ref_idx is not None and self._ref_idx >= len(self._lights):
            self._ref_idx = None
        if self._lights:
            self._load_reference_preview(force=False)
        else:
            self._ref_idx = None
            self.lbl_ref.setText("Référence : — (ajoutez des images)")

    def _ref_index(self) -> int:
        if not self._lights:
            return 0
        if self._ref_idx is None:
            return len(self._lights) // 2  # défaut : image du milieu
        return min(self._ref_idx, len(self._lights) - 1)

    def _highlight_ref(self) -> None:
        """Met en gras la ligne de la référence dans la liste des images."""
        idx = self._ref_index()
        for r in range(self.list_lights.count()):
            it = self.list_lights.item(r)
            w = self.list_lights.itemWidget(it)
            lab = w.findChild(QLabel) if w is not None else None
            if lab is not None:
                f = lab.font()
                f.setBold(r == idx)
                lab.setFont(f)

    def _on_light_clicked(self, item) -> None:
        """Clic sur une image de la liste -> elle devient la RÉFÉRENCE (image de base)."""
        row = self.list_lights.row(item)
        if not (0 <= row < len(self._lights)):
            return
        self._ref_idx = row
        idx = self._ref_index()
        self._log(
            f"Référence changée par clic : {os.path.basename(self._lights[idx])} "
            f"(#{idx + 1}/{len(self._lights)}) — aperçu rechargé, masque réinitialisé."
        )
        self._load_reference_preview(force=True)

    # ---------- aperçu de la référence ----------
    def _load_reference_preview(self, force: bool = False) -> None:
        if not self._lights:
            return
        idx = self._ref_index()
        self.lbl_ref.setText(f"Référence : {os.path.basename(self._lights[idx])}")
        self._highlight_ref()
        if not force and self.mask_editor.has_image():
            m = self.mask_editor.get_mask()
            if m is not None and m.any():
                return  # un masque a déjà été peint : on ne change pas l'aperçu
        self._log(
            f"Référence : {os.path.basename(self._lights[idx])} "
            f"(#{idx + 1}/{len(self._lights)}) — aperçu chargé."
        )
        QApplication.processEvents()
        try:
            img = load_image_linear(self._lights[idx]).astype("float32")
        except Exception as exc:
            self._log(f"ERREUR lecture référence : {exc}")
            return
        h, w = img.shape[:2]
        # L'aperçu de la référence TIENT DANS LA FENÊTRE : on le réduit pour entrer dans
        # la zone visible (plafond 1200 px) ; la barre de déroulement intervient si la
        # fenêtre est ensuite plus petite. Le masque peint est re-redimensionné à la
        # pleine résolution par le pipeline, donc peindre sur l'aperçu réduit est sûr.
        vp = self.mask_scroll.viewport().size()
        scale = min(1.0, 1200.0 / max(h, w))
        if vp.width() > 60 and vp.height() > 60:
            scale = min(scale, (vp.width() - 10) / w, (vp.height() - 10) / h)
        if scale < 1.0:
            import cv2

            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        self.mask_editor.set_image(u16_to_qimage(img))
        self.tabs.setCurrentIndex(0)
        self._log("Peignez l'AVANT-PLAN (sol, arbres…) : il restera fixe. Le ciel non peint sera aligné.")

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier de sortie", self._out_dir)
        if d:
            self._out_dir = d
            self.edit_out.setText(self._out_dir)

    def _on_erase_toggled(self, v: bool) -> None:
        self.mask_editor.erase = v
        self._log("Gomme activée (efface le masque peint)" if v else "Pinceau (peint l'avant-plan)")

    def _bump_brush(self, delta: int) -> None:
        """Change la taille du pinceau (synchronisée avec le slider « Pinceau»)."""
        v = min(self.slider_brush.maximum(), max(self.slider_brush.minimum(), self.slider_brush.value() + delta))
        self.slider_brush.setValue(v)
        self._log(f"Pinceau : {v} px")

    def eventFilter(self, obj, ev):
        """Molette de la souris sur l'aperçu de la référence -> pinceau +/- 10 px."""
        if ev.type() == QEvent.Wheel and obj in (self.mask_editor, self.mask_scroll.viewport()):
            delta = ev.angleDelta().y()
            if delta:
                self._bump_brush(+10 if delta > 0 else -10)
                return True  # événement consommé : la molette ne fait pas défiler la vue
        return super().eventFilter(obj, ev)

    def _show_shortcuts(self) -> None:
        """F1 : affiche les raccourcis clavier (boîte de dialogue + log)."""
        rows = "<br>".join(f"<b>{k}</b> — {t}" for k, (t, _fn) in self._shortcuts.items())
        QMessageBox.information(self, "Raccourcis clavier", rows + "<br><b>F1</b> — cette aide")
        self._log("Raccourcis : " + " · ".join(f"{k} = {t}" for k, (t, _fn) in self._shortcuts.items()) + " · F1 = aide")

    def _log(self, msg):
        self.log.appendPlainText(msg)

    # ---------- masque ----------
    def _reset_lum_params(self) -> None:
        """v0.8.18 — bouton « Reset » : remet les paramètres de LUMINANCE aux valeurs par
        défaut (« Affiner la luminance » cochée, Seuil 0 = auto Otsu, Bande 40 px,
        Rés. 2000 px)."""
        self.chk_lum_refine.setChecked(True)
        self.spin_lum_thr.setValue(0)
        self.spin_edge_band.setValue(40)
        self.spin_mask_res.setValue(2000)
        self._log("Luminance : paramètres remis par défaut (case cochée, Seuil auto, Bande 40 px, Rés. 2000 px).")

    def _ml_mask(self):
        if not self.mask_editor.has_image():
            QMessageBox.warning(self, "Masque ML", "L'aperçu de la référence n'est pas chargé (ajoutez des lights).")
            return
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            QMessageBox.information(
                self,
                "Masque ML",
                "Le masque ML a besoin de onnxruntime :\n\n    pip install onnxruntime\n\n"
                "Relancez ensuite. Le masque par seuil (bouton « Auto (seuil) ») reste disponible.",
            )
            return
        from .imaging import tonemap_u16
        from .mlmask import sky_to_foreground_mask

        self._log("Masque ML : téléchargement du modèle ONNX si besoin, puis inférence sur la RÉFÉRENCE "
                  f"(plafond {self.spin_mask_res.value()} px, comme en v0.6.0)…")
        QApplication.processEvents()
        # Le modèle voit la RÉFÉRENCE à pleine résolution utile — PAS l'aperçu
        # réduit à la taille de la fenêtre : sur une petite image le modèle répond « 100 % ciel ».
        ref_path = self._lights[self._ref_index()]
        try:
            img = load_image_linear(ref_path).astype("float32")
        except Exception as exc:
            self._log(f"ERREUR lecture de la référence pour le masque ML : {exc}")
            return
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        h, w = img.shape[:2]
        max_side = self.spin_mask_res.value()  # v0.8.14 : résolution de travail réglable (défaut 2000 px)
        if max(h, w) > max_side:
            import cv2

            s = max_side / max(h, w)
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        img_u8 = tonemap_u16(img)
        fg = sky_to_foreground_mask(
            img_u8,
            debug_cb=self._log,
            top_gradient=self.chk_top_sky.isChecked(),
            top_boost=self.spin_top_boost.value(),
            grow=0,  # v0.8.12 : plus d'élargissement +2 px du masque ML (il rognait le ciel)
            lum_refine=self.chk_lum_refine.isChecked(),  # v0.8.11 : affinage luminance ancré (sombre -> avant-plan, lueur -> ciel)
            lum_thr=(self.spin_lum_thr.value() or None),  # v0.8.14 : seuil manuel 0..255 (0 = Otsu auto)
            lum_edge=self.spin_edge_band.value() > 0,     # v0.8.14 : bord recalé sur la luminance pleine résolution
            edge_band=self.spin_edge_band.value(),        # v0.8.14 : largeur de la bande (px)
        )
        self.mask_editor.set_mask(fg)  # set_mask redimensionne le masque à l'aperçu affiché
        self._log("Masque ML appliqué — retouchez au pinceau : rouge = avant-plan (fixe), le reste sera aligné.")

    # ---------- lancement ----------
    def _unique_out(self):
        base = os.path.join(self._out_dir, "astrostack_result.tif")
        if not os.path.exists(base):
            return base
        i = 2
        while os.path.exists(os.path.join(self._out_dir, f"astrostack_result_{i}.tif")):
            i += 1
        return os.path.join(self._out_dir, f"astrostack_result_{i}.tif")

    def _run(self):
        if not self._lights:
            QMessageBox.warning(self, "Empilement", "Ajoutez des lights.")
            return
        if self._worker is not None and self._worker.isRunning():
            return
        mask = self.mask_editor.get_mask() if self.mask_editor.has_image() else None
        scale = [1.0, 0.5, 0.25][self.combo_scale.currentIndex()]
        mode = ["sigma", "mean"][self.combo_mode.currentIndex()]
        ref_index = self._ref_index()
        out_path = self._unique_out()
        cfg = dict(
            lights=self._lights,
            darks=self._darks or None,
            flats=self._flats or None,
            biases=self._biases or None,
            mask=mask,
            scale=scale,
            mode=mode,
            kappa=self.spin_kappa.value(),
            refine_ecc=self.chk_ecc.isChecked(),
            refine_local=self.chk_local.isChecked(),
            ref_index=ref_index,
            out_path=out_path,
            lp_correct=self.chk_lp.isChecked(),
            lp_method=["synthetic", "graxpert"][self.combo_lp.currentIndex()],
            lp_strength=self.spin_lp.value() / 100.0,
            lp_sky_only=self.chk_lp_sky.isChecked(),
            lp_feather=self._lp_feather_value(),  # v0.8.15 : fondu PL réglable en UI (« auto » par défaut)
            lp_mask_source=["painted", "ml"][self.combo_lp_mask.currentIndex()],
            meta_out=self._meta,   # v0.8.4 : sky/fg_w/stack_raw pour réappliquer la PL
        )
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self._log(
            f"Empilement de {len(self._lights)} lights (référence : "
            f"{os.path.basename(self._lights[ref_index])}, #{ref_index + 1}) -> {out_path}"
        )
        self._worker = StackWorker(cfg, self)
        self._worker.progressed.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _on_progress(self, stage, i, n, msg):
        if n:
            self.progress.setMaximum(n)
            self.progress.setValue(i)
        if msg:
            self._log(msg)

    def _on_done(self, result):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        raw = self._meta.get("stack_raw")  # v0.8.4 : stack AVANT correction PL
        self._stack_base = raw if raw is not None else result
        self.btn_lp_redo.setEnabled(self._stack_base is not None)
        self.tabs.setCurrentIndex(1)
        qimg = u16_to_qimage(result)
        self._result_full = QPixmap.fromImage(qimg)  # pleine résolution conservée
        self._fit_result()
        self._log("Terminé ! TIFF 16 bits + aperçu PNG + masque PNG écrits dans le dossier de sortie.")

    def _lp_feather_value(self):
        """v0.8.15 — spin « Fondu » de la box PL : -1 (« auto ») -> None (sigma auto = min(h,l)/96)."""
        v = self.spin_lp_feather.value()
        return None if v < 0 else float(v)

    def _on_lp_method_changed(self, idx: int) -> None:
        """v0.8.10 : « ML GraXpert » corrige TOUTE l'image (le modèle ONNX protège
        l'avant-plan, y compris la lueur d'horizon) -> « Ciel uniquement » est grisé et
        décoché pour GraXpert ; réactivé pour « Synthetic flat ».
        v0.8.11 : « Masque PL » est aussi grisé en GraXpert (inutile : correction globale)."""
        graxpert = idx == 1
        self.chk_lp_sky.setEnabled(not graxpert)
        if graxpert:
            self.chk_lp_sky.setChecked(False)
        self.chk_lp_sky.setToolTip(
            "N'applique la correction que sur le ciel (masque avant-plan) — utile avec "
            "« Synthetic flat » (anti-halos). Grisé en « ML GraXpert » : GraXpert corrige "
            "TOUTE l'image, le modèle protège l'avant-plan, donc la lueur d'horizon est corrigée aussi."
        )
        self.combo_lp_mask.setEnabled(not graxpert)  # v0.8.11 : masque PL sans objet en GraXpert (correction globale)
        self.combo_lp_mask.setToolTip(
            "Masque utilisé POUR la correction de la pollution lumineuse : « Peint (UI) » ou « ML ONNX (auto) » "
            "(ONNX brut, v0.8.9). Grisé en « ML GraXpert » (v0.8.11) : GraXpert corrige TOUTE l'image, aucun masque PL."
        )

    def _redo_lp(self) -> None:
        """v0.8.4 — Réapplique la correction PL avec les réglages actuels sur le dernier
        stack terminé, SANS refaire l'alignement ni l'empilement : sauve un nouveau TIFF."""
        if self._stack_base is None:
            QMessageBox.warning(self, "Correction PL", "Aucun empilement terminé pour le moment.")
            return
        from .lpcorrect import correct_light_pollution
        from .pipeline import _save_result

        img = self._stack_base
        if ["painted", "ml"][self.combo_lp_mask.currentIndex()] == "ml":  # v0.8.5 : masque ONNX pour la PL
            import cv2

            from .mlmask import foreground_mask_from_linear
            from .rawio import load_image_linear
            from .stacker import feather

            ref_path = self._meta.get("ref_path") or (self._lights[self._ref_index()] if self._lights else None)
            wsh = self._meta.get("work_shape") or img.shape[:2]
            fg = foreground_mask_from_linear(load_image_linear(ref_path).astype(np.float32),
                                             debug_cb=self._log, top_gradient=False, top_boost=0.5,
                                             grow=0, smooth=1, proba_blur=0, morph=False)  # v0.8.7/0.8.9 : ONNX 100 % brut pour la PL
            fg = cv2.resize(fg, (wsh[1], wsh[0]), interpolation=cv2.INTER_NEAREST)
            sky = 1.0 - feather(fg, sigma=max(2.0, min(wsh) * 0.004))
            self._log("Masque PL : ONNX brut (seuil 0,5, sans lissage ni dilatation).")
        else:
            sky = self._meta.get("sky")
        fg_w = self._meta.get("fg_w")
        method = ["synthetic", "graxpert"][self.combo_lp.currentIndex()]
        strength = self.spin_lp.value() / 100.0
        sky_only = self.chk_lp_sky.isChecked()
        feather_v = self._lp_feather_value()  # v0.8.15 : fondu PL réglable
        out_path = self._unique_out()
        if sky is not None:  # v0.8.9 : sauvegarde du masque PL utilisé (diagnostic)
            try:
                import imageio.v3 as iio

                b0, _ = os.path.splitext(out_path)
                iio.imwrite(b0 + "_maskPL.png", (np.clip(sky, 0.0, 1.0) * 255.0).round().astype(np.uint8))
                self._log(f"Masque PL écrit : {b0}_maskPL.png — ciel = {float((sky > 0.5).mean()):.0%} de l'image")
            except Exception:
                pass
        self._log(
            f"Réapplication PL — méthode « {method} », force {strength:.0%}, "
            f"ciel uniquement={sky_only} — sur le dernier stack -> {out_path}"
        )
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            if self.chk_lp.isChecked():
                out = correct_light_pollution(
                    img, method=method, strength=strength, sky=sky, sky_only=sky_only,
                    feather=feather_v,  # v0.8.15
                    debug_cb=self._log,
                )
            else:
                out = img  # case PL décochée : on sauvegarde le stack tel quel
                self._log("Correction PL désactivée : sauvegarde du stack tel quel.")
            if fg_w is None:
                fg_w = np.zeros(out.shape[:2], np.float32)
            paths = _save_result(out_path, out, fg_w)
            qimg = u16_to_qimage(out)
            self._result_full = QPixmap.fromImage(qimg)
            self._fit_result()
            self.tabs.setCurrentIndex(1)
            self._log("Réapplication terminée : " + " ; ".join(f"{k} = {v}" for k, v in paths.items()))
        except Exception as exc:  # noqa: BLE001
            self._log(f"ERREUR correction PL : {type(exc).__name__} : {exc}")
            QMessageBox.critical(self, "Erreur", f"{type(exc).__name__} : {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _fit_result(self):
        """L'aperçu du résultat tient dans la fenêtre : il est redimensionné à la zone
        visible (la barre de déroulement intervient si la fenêtre est plus petite)."""
        pix = self._result_full
        if pix is None or pix.isNull():
            return
        vp = self.result_scroll.viewport().size()
        if vp.width() < 10 or vp.height() < 10:
            return
        scaled = pix.scaled(vp.width() - 4, vp.height() - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_label.setPixmap(scaled)
        self.result_label.resize(scaled.size())

    def _on_failed(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log("ERREUR : " + msg)
        QMessageBox.critical(self, "Erreur", msg)

    def _on_stop(self) -> None:
        """Bouton ⏹ Arrêter : annulation coopérative de l'empilement en cours."""
        if self._worker is not None and self._worker.isRunning():
            self._log("Arrêt demandé — annulation de l'empilement…")
            self._worker.cancel()
            self.btn_stop.setEnabled(False)

    def _on_cancelled(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log("Empilement annulé par l'utilisateur.")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # rendu identique Mac/Windows, base du thème sombre
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
