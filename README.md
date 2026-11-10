# 🌌 AstroStack

Empilement de paysages de nuit (*star stacking*) pour **macOS et Windows**, inspiré de
**Sequator**, **Starry Landscape Stacker** et **SharpStacker** :

- lecture des **RAW (CR2, CR3, DNG, NEF, ARW, ORF, RW2, RAF, PEF, SRW)** via LibRaw, et des TIFF/PNG/JPEG ;
- **alignement des étoiles** sur la 1ère image : corrélation de phase (grande dérive) puis appariement d'étoiles et **homographie RANSAC**, affinée par **ECC photométrique** sur le ciel, puis **warp local par thin-plate spline (TPS)** à partir des étoiles appariées réparties sur tout le champ — c'est ce qui corrige la **distorsion de l'objectif** et les traînées dans les **coins** (désactivable : case « Warp local (TPS) » / `--no-local`, ECC : `--no-ecc`) ;
- **avant-plan fixe** : vous peignez l'avant-plan (sol, arbres…) avec un **pinceau** comme dans un outil de retouche — le ciel est recalé et empilé, l'avant-plan reste net et empilé sans bouger, avec un raccord en fondu ;
- prise en compte des **darks** (master dark = moyenne), plus **flats** et **bias** optionnels —
  quand les lights et les darks sont des RAW, la soustraction se fait **sur le Bayer brut, avant
  dématriçage** (via `raw_image_visible`), avec gestion du black level par canal ;
- **masque ML (ONNX)** comme SharpStacker : bouton « Masque ML (ONNX) » qui segmente le ciel avec un
  SegFormer-B0 ADE20K en ONNX (téléchargé automatiquement depuis Hugging Face, ~14 Mo, mis en cache
  dans `~/.astrostack/`) — **probabilité softmax du canal « sky » remontée en cubique + lissée** (bords
  nets, sans blocs) — le masque reste **éditable au pinceau** (repeinture, gomme, seuil, inverser) ;
  en CLI : `--auto-mask ml` (nécessite `pip install onnxruntime`, extra `ml`) ;
- **sigma clipping** pour rejeter avions, satellites, lucioles ;
- export **TIFF 16 bits** (linéaire) + aperçu PNG + masque PNG.

Langage : **Python** (choisi pour la maturité de `rawpy`/LibRaw et d'OpenCV, et PySide6 pour une vraie GUI
cross-platform Mac/Windows identique). Le cœur (alignement + stacking) est isolé du Qt et tourne aussi en CLI.

## Installation

```bash
python -m venv .venv
# macOS / Linux :
source .venv/bin/activate
# Windows :
.venv\Scripts\activate

pip install -r requirements.txt
```

> `rawpy` fournit des wheels précompilés (LibRaw inclus) pour macOS (Apple Silicon & Intel) et Windows.

## Utilisation (GUI)

```bash
python run.py
```

1. **Ajouter** les lights (RAW) — la 1ère sert de référence — puis les **darks** (ISO/exposition/température proches des lights).
2. Cliquez sur **« Peindre le masque sur la référence »** : peignez l'**avant-plan** en rouge (pinceau réglable, gomme, auto-seuil, inverser, effacer).
3. **« Lancer l'empilement »** → `astrostack_result.tif` (16 bits) + `_preview.png` + `_mask.png` dans le dossier de sortie.

## Utilisation (CLI, sans GUI)

```bash
python -m astrostack.cli --lights-dir ./lights --darks-dir ./darks \
    --mask masque.png --out resultat.tif --mode sigma --kappa 2.5
```

(`--mask` : PNG 8 bits, 255 = avant-plan. Sans `--mask`, tout le ciel est aligné/empilé.)

## Comment ça marche

```
RAW/TIFF ──► calibration (bias? dark! flat?) ──► RGB linéaire 16 bits
                                                        │
                       ┌────────────────────────────────┘
                       ▼
              détection d'étoiles (zone ciel = hors masque)
                       │
                       ▼
     alignement : corrélation de phase (grande dérive) puis appariement d'étoiles
     + HOMOGRAPHIE RANSAC + affinage ECC sur le ciel (échelle réduite, puis
     remise à pleine résolution)  ──► homographie 3x3 frame→référence
                       │
                       ▼
              warp du ciel + sigma clipping + moyenne ; avant-plan empilé tel quel
                       │
                       ▼
              composition  =  ciel_recalé * (1-masque)  +  avant_plan * masque   (masque flouté)
                       │
                       ▼
                 TIFF 16 bits
```

- Le masque peint est redimensionné à la résolution de travail puis **flouté** (feathering) pour un raccord sans couture.
- **Toutes les images sont automatiquement redimensionnées sur la 1ère light** (la référence, dans l'ordre de la liste) : lights, darks, flats, bias et masque. Un fichier de taille différente (autre appareil, JPEG, etc.) ne fait plus planter le programme. Si un fichier est illisible, le message d'erreur nomme le fichier concerné.
- Option **résolution de travail** (100/50/25 %) pour limiter la RAM sur les gros capteurs.
- Le pipeline tient en RAM ~2 arrays float32 pleine résolution + cache de frames limité à ~2,5 Go ; sinon les RAW sont relus à la 2ème passe.

## Test

```bash
pip install numpy opencv-python-headless imageio tifffile   # pas besoin de rawpy ni Qt
python tests/test_synthetic.py
```

Génère une scène synthétique (ciel étoilé + sol), 6 poses bruitées/décalées + un dark, et vérifie
l'alignement, le repli par corrélation de phase, la soustraction du dark, la réduction du bruit et la
fixité de l'avant-plan.

## Télécharger l'application (sans Python)

Les versions prêtes à installer sont publiées dans la page **Releases** du dépôt :

- **macOS (Apple Silicon)** : `AstroStack-<version>-macos-arm64.dmg` → ouvrir, glisser `AstroStack` dans `Applications`.
  L'application n'est pas encore signée par Apple : au premier lancement, faire un clic droit sur l'app → **Ouvrir**
  (ou : `xattr -dr com.apple.quarantine /Applications/AstroStack.app`).
- **Windows (64 bits)** : `AstroStack-Setup-<version>.exe` → lancer l'installeur. SmartScreen peut afficher un
  avertissement (« éditeur inconnu ») : **Informations complémentaires → Exécuter quand même**.

Le masque ML (ONNX) est inclus ; son modèle (~14 Mo) est téléchargé au premier usage.

## Publier une nouvelle version (build automatique)

1. Mettre à jour la version dans `astrostack/__init__.py` (`__version__`) et `pyproject.toml`.
2. Créer et pousser un tag : `git tag v1.2.0 && git push origin v1.2.0`.
3. GitHub Actions (`.github/workflows/release.yml`) compile l'application sur macOS et Windows, puis crée la
   Release avec le `.dmg` et l'installeur `.exe`. Le même workflow peut être lancé à la main depuis
   l'onglet **Actions** (sans créer de Release, les fichiers sont dans les artefacts du run).

L'icône est générée par `packaging/make_icons.py` (source : `packaging/icon.png`, redessinable avec `--draw`).

### Compiler localement

```bash
pip install -r requirements.txt onnxruntime pyinstaller
python packaging/make_icons.py
pyinstaller --noconfirm packaging/astrostack.spec
# macOS : dist/AstroStack.app   Windows : dist/AstroStack/AstroStack.exe
# (installeur Windows : ISCC /DAppVersion=1.2.0 packaging\astrostack.iss)
```

## Historique

Le détail de chaque version est dans [CHANGELOG.md](CHANGELOG.md).

## Limites connues (v1) et pistes v2

- Si la référence est un RAW mais qu'un fichier de calibration est un TIFF/PNG, tout repasse en calibration post-dématriçage (log : message explicite).
- Le masque ML segmente la classe « sky » d'ADE20K (indice 2) : sur des scènes de nuit très sombres, prévoir à retoucher au pinceau (c'est fait pour). Modèle surchargeable via `ASTROSTACK_ONNX_MODEL` / `ASTROSTACK_ONNX_URL`.
- Pas encore de : mode comète, star trails, export FITS. Le CLI est pensé pour ça.
- Rust/C++ : le cœur `stars.py` + `stacker.py` se traduit directement (OpenCV a des bindings Rust/C++) si besoin de performances/exécutable natif unique.

## Structure

```
astrostack/
  rawio.py       lecture RAW (rawpy) ; calibration Bayer brut (dark/bias/flat avant dématriçage)
                 ou post-dématriçage ; sortie RGB linéaire 16 bits
  stars.py       détection d'étoiles, corrélation de phase, appariement (ratio de Lowe),
                 homographie RANSAC + affinage ECC + warp local TPS (distorsion), remise
                 à pleine résolution
  stacker.py     empilement par bandes de lignes : ciel recalé + sigma clipping,
                 avant-plan fixe, raccord fondu
  lpcorrect.py   correction de la pollution lumineuse (fond polynomial robuste, forme commune)
  pipeline.py    run_pipeline() : orchestration complète, export TIFF 16 bits + aperçus
  mlmask.py      masque ML ONNX (SegFormer ADE20K, sky=classe 2), téléchargeable
  imaging.py     tonemapping pour les aperçus
  preview.py     conversions d'image pour l'affichage Qt
  cli.py         ligne de commande (--auto-mask heuristic|ml)
  mainwindow.py  fenêtre principale (Qt) : panneaux, actions, aperçus
  maskwidget.py  éditeur de masque au pinceau (Qt)
  zoomview.py    zoom et centrage des aperçus (Qt)
  widgets.py     widgets réutilisables (sections repliables)
  worker.py      thread de calcul (Qt), annulable
  theme.py       feuille de style sombre (Qt)
tests/           tests unitaires et synthétiques (python -m pytest tests)
CHANGELOG.md     historique des versions
```
