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

## Packaging en application (Mac / Windows)

```bash
pip install pyinstaller
pyinstaller --windowed --name AstroStack run.py
# macOS : dist/AstroStack.app   Windows : dist/AstroStack/AstroStack.exe
```

## Historique

- **v1.0.0** — **pollution lumineuse réécrite** : le ciel est détecté par le **masque automatique** (ONNX affiné) calculé sur le stack ; fond estimé par **ajustement polynomial robuste** à forme commune aux trois couleurs ; cible = **médiane du ciel** ; plancher doux ; transition douce au bord ; force bornée à 100 % ; fond estimé exporté (`<base>_fondPL.png`). **Méthode GraXpert retirée** (licence non commercial). Le masque peint ne sert plus à la correction. CLI : `--lp-method`, `--no-lp-sky-only` et `--lp-mask` supprimés ; `--lp-correct`, `--lp-strength`, `--lp-feather`, `--lp-redo` conservés. Ancien flou gaussien retiré.
- **v0.9.3** — **pollution lumineuse améliorée** : (1) fond estimé par **ajustement polynomial robuste** (forme du gradient partagée par les trois couleurs, étoiles rejetées) au lieu du flou gaussien — résidu de gradient 220/156/93 → 0 à 11 sur scène synthétique ; `bg_method="blur"` garde l'ancien ; (2) cible = **médiane du ciel** (le niveau du ciel n'est plus remonté au percentile 10) ; (3) **plancher doux** au lieu d'un écrêtage à zéro ; (4) un masque déjà adouci n'est plus re-flouté ; (5) **force bornée à 100 %** (interface) ; (6) **fond estimé exporté** (`<base>_fondPL.png`) pour contrôle visuel ; (7) `tests/test_lp.py`.
- **v0.9.2** — **marge de 2 px pour l'empilement** (`stack_grow=2`) : le masque d'avant-plan utilisé pour **empiler** est élargi de 2 px, pour que les bords de l'arbre viennent de la référence et non des poses warpées (fantômes autour des arbres). Cette marge n'apparaît **ni** dans le masque affiché/exporté (`_mask.png`), **ni** dans la correction de pollution lumineuse.
- **v0.9.1** — **option expérimentale `--mode robust`** (interface : non exposée) : rejet du ciel centré sur la **médiane** avec écart robuste (MAD), puis moyenne des valeurs conservées, quand toutes les poses tiennent en mémoire (~1,5 Go). Le mode par défaut `sigma` reste inchangé. Sur scène synthétique de trépied (sol fixe, ciel qui dérive), le fantôme autour de l'arbre passe de 112 à 43 ; ce gain n'a pas été confirmé sur de vraies photos.
- **v0.9.0** — **masque automatique plus précis** (`--auto-mask ml`, bouton « Masque ML ») : le bord du masque ONNX est d'abord recalé sur la luminance pleine résolution avec un **seuil de luminance local** (moyenne des luminances avant-plan / ciel autour de chaque pixel, au lieu d'un seul seuil Otsu global : meilleur sur avant-plans très sombres et lueur d'horizon), puis lissé par un **filtre guidé** sur la luminance pleine résolution. Le lissage médian et l'ouverture morphologique, qui effaçaient les branches fines, ne sont plus appliqués sur ce chemin. Le masque reste souple au bord. En réduction de résolution, le masque est re-affiné à pleine résolution au lieu d'un agrandissement au plus proche. Mesuré sur scènes synthétiques (grille modèle 16 px) : IoU 0,952 → 0,985 sur un ciel sombre bruité avec branches de 1 px, et sans débordement de l'avant-plan dans le ciel (bord à ±2 px de la vérité). Le seuil de luminance ne décide que si le contraste local est net. CLI : `--no-mask-guided` pour l'ancien comportement. Le masque de correction de pollution lumineuse n'est pas modifié.
- **v0.8.18** — **Bouton « Reset » pour les paramètres de luminance** (box Masque, à droite de Seuil/Bande/Rés.) : remet « Affiner la luminance » cochée + **Seuil 0 (auto) / Bande 40 px / Rés. 2000 px** (méthode `_reset_lum_params`, logué). Base = fichier utilisateur repris tel quel (`mainwindow-6.py`) : flèches **▼/▶**, « Darks » repliée par défaut, bouton « Masque ML (ONNX)… » déplacé sous la ligne luminance, libellé « Affiner la luminance », boîte « **Stacking Options** », panneau **360 px**.
- **v0.8.17** — **Panneau gauche : aucune barre de scroll horizontale possible** (l'utilisateur la voyait encore en bas malgré `ScrollBarAlwaysOff`) : en plus de la policy off, la scrollbar elle-même est cachée (`horizontalScrollBar().setVisible(False)`), un **catch-all `findChildren(QAbstractScrollArea)`** applique ça au log et aux listes, une **QSS `QScrollBar:horizontal { height: 0px; }`** sur `panel_scroll` rend toute barre horizontale invisible (0 px) même si elle est « shown », et `panel.setMinimumWidth(0)` + `combo_lp.setMinimumWidth(0)` garantissent que le contenu tient dans les 340 px. Seule la barre **verticale** peut encore apparaître (`AsNeeded`) si le contenu dépasse en hauteur.
- **v0.8.16** — **Plus aucune barre de scroll horizontale dans le panneau gauche (340 px)** : le log revient à la ligne (`WidgetWidth`) avec `ScrollBarAlwaysOff`, les listes de fichiers ont `ScrollBarAlwaysOff` + `ElideMiddle` (noms longs tronqués « … »), `panel_scroll` reste en `ScrollBarAlwaysOff` (seule la barre **verticale** peut apparaître en cas de contenu trop haut) ; la ligne « Seuil / Bande / Rés. » est resserrée (marges 0, spins 46/46/62 px) et `combo_lp_mask.setMinimumWidth(0)` pour ne jamais déborder en largeur.
- **v0.8.15** — **« Fondu » de la correction PL exposé en UI** : nouveau spin **« Fondu »** dans la box « Pollution lumineuse » (ligne « Masque PL ») — `auto` (défaut, = `min(h,l)/96`), `0` = masque dur, ou sigma en px 1–200 — c'est le paramètre anti-halo autour de l'avant-plan (ex. balle), auparavant réservé au CLI (`--lp-feather`). Transmis au pipeline **et** au bouton « Réappliquer la PL » (méthode `_lp_feather_value`). Aucun autre changement : le masque auto reste calé sur Seuil / Bande / Rés. (v0.8.14).
- **v0.8.14** (base **v0.8.12** — le tuilage v0.8.13 est abandonné : le SegFormer ADE20K a une frontière de classe trop grossière pour les branches, le tuilage n'ajoutait que des artefacts dans le ciel) — **pack « détail du bord »**, 3 réglages en **une seule ligne compacte** dans la box « Masque d'avant-plan » (le panneau 340 px est préservé) : **« Bande »** (0–200 px, défaut **40**) — dans une bande de ±N px autour du bord du masque ONNX, le bord est **recalé sur la luminance pleine résolution** (`lum <= seuil`) : les **fines branches sombres sur ciel lumineux** sont détectées à la résolution de l'image, et le ciel loin du bord reste intact (pas d'artefacts) ; **« Seuil »** (0–255, défaut 0 = **Otsu auto**) — seuil de luminance réglable ; **« Rés. »** (800–3200 px, défaut **2000**) — **résolution de travail du masque ML** (1200 px avant), pour des affinages plus fins. CLI : `--mask-edge-band N` (0 = off), `--mask-lum-thr N`, `--mask-max-side N` (défaut 2000). L'ONNX garde la forme globale, la luminance gère le détail du bord.
- **v0.8.12** — **Masque ML d'empilement : précision et granularité** : (1) **suppression de l'élargissement +2 px** (`grow` 2 → **0** par défaut, UI + `--auto-mask ml`) — il rognait le ciel au bord des objets ; (2) **affinage luminance ancré sur le masque ONNX de base** — un pixel sombre ne devient avant-plan (et une zone lumineuse ciel) que s'il est **relié** (dilatation ~2,5 % du petit côté) au blob ONNX correspondant : **fini les îlots « sans lien »** (ex. coins vignettés sombres classés avant-plan en haut de l'image, visibles en rouge sur l'éditeur) ; (3) **granularité** : lissage **médian 5×5** du masque binaire (efface les bords en escalier du sur-échantillonnage 512 px de la proba ONNX) + **suppression des petites composantes connexes isolées** (< 0,15 % de l'image, min 150 px). Le masque PL (correction) reste brut, inchangé.
- **v0.8.11** — **Masque ML d'empilement affiné par la luminance** : nouvelle case « **Affiner avec la luminance** » (cochée par défaut, box Masque) — après le masque ONNX, les pixels **sombres** (seuil **Otsu automatique** sur la luminance Rec.709) sont forcés en **avant-plan**, et les grandes zones **très lumineuses** (seuil × 1,5, nettoyées par ouverture morphologique 5×5 pour éliminer fenêtres et étoiles isolées) basculent en **ciel** — typiquement la **lueur d'horizon** que le SegFormer classe « avant-plan » (visible sur `maskPL.png`). CLI : `--no-mask-lum` pour désactiver, `--mask-lum-thr N` (0..255, 0 = auto Otsu). **En « ML GraXpert », le combo « Masque PL » est désormais aussi grisé** (correction globale, aucun masque PL) et le **contraste des widgets désactivés est renforcé** (texte `#6f6f7a` sur fond `#222228`, au lieu d'un gris à peine visible).
- **v0.8.10** — **Diagnostic confirmé par le masque** : la lueur d'horizon est classée « avant-plan » (noire) par le masque ONNX, donc « Ciel uniquement » la laissait non corrigée → gradient résiduel en bas. **Méthode « ML GraXpert » = correction sur TOUTE l'image** : la case « Ciel uniquement » est automatiquement grisée et décochée en GraXpert (UI), et le pipeline force `sky_only=False` pour GraXpert (CLI inclus) — le modèle ONNX GraXpert protège lui-même l'avant-plan (objets) tout en corrigeant la lueur d'horizon. En « Synthetic flat », « Ciel uniquement » reste disponible (anti-halos).
- **v0.8.9** — Audit du chemin PL + **masque ONNX 100 % brut** : depuis v0.8.7 aucun dégradé vertical ni dilatation n'étaient appliqués à la PL, mais il restait un lissage de la proba (σ=1), une morphologie 3×3 et un smooth 3×3 sur le masque — supprimés pour la PL (`proba_blur=0, morph=False, smooth=1, grow=0, top_gradient=False` ; seuls le seuil 0,5 de la proba « ciel » et le fondu de bord du pipeline restent). **Diagnostic** : le masque PL réellement utilisé est sauvegardé en **`<sortie>_maskPL.png`** (pipeline, « Réappliquer la PL » et `--lp-redo`), avec le % de ciel logué — si la lueur d'horizon y est noire (classée avant-plan par l'ONNX), c'est elle qui garde le gradient avec « Ciel uniquement » cochée (décochez-la ou repeignez au-dessus de la lueur). Côté image, aucun dégradé n'est jamais ajouté : le tonemap de l'aperçu est un étirement global de percentiles + gamma, et le seul « lissage » est l'estimation du fond elle-même.
- **v0.8.8** — **Boîtes repliables : flèche ▾ (ouverte) / ▸ (fermée)** dans le titre à la place de la case à cocher (indicateur masqué en QSS). **`astrostack_result_avant_PL.tif` écrit une seule fois** : si le fichier existe déjà, il est conservé tel quel (fini la réécriture à chaque test de la correction PL). **Fond « synthetic flat » plus serré** (`max_small` 512→**768**, sigma `min/8`→**min/12**) : capture mieux la lueur d'horizon (gradient résiduel) tout en lissant étoiles et Voie lactée. Note : si le masque ML classe la lueur d'horizon comme avant-plan, « Ciel uniquement » la laisse non corrigée — décochez « Ciel uniquement » ou repeignez le masque au-dessus de la lueur.
- **v0.8.7** — **« Masque PL = ML ONNX (auto) » utilise le masque ONNX BRUT** pour la correction de la pollution lumineuse : **plus de dilatation +2 px** (`grow=0`) **ni de dégradé vertical** « haut = ciel » (`top_gradient=False`) — ces réglages restent réservés au masque d'**empilement** (où `grow=2` et le dégradé continuent de s'appliquer). Appliqué dans le pipeline, dans « Réappliquer la PL » (UI) et dans `--lp-redo --lp-mask ml` (CLI).
- **v0.8.6** — **Anti-halo (fin)** : le poids de fusion ciel/avant-plan est dilaté de **1,5·sigma** avant lissage (au lieu de 0,5·sigma en v0.8.5) — avec un lissage symétrique le bord ciel ne recevait qu'environ 70 % de la correction, laissant ~30 % du gradient résiduel collé à l'avant-plan (halo persistant autour de la balle) ; à 1,5·sigma le bord ciel reçoit ~93 % de la correction et la transition passe quasiment entièrement côté avant-plan. **Force de la correction PL jusqu'à 150 %** dans l'UI (sur-correction possible pour tuer un résidu ; CLI `--lp-strength` accepte déjà toute valeur).
- **v0.8.5** — **Anti-halo (suite)** : le fond « synthetic flat » n'est estimé que sur du **ciel « pur »** (érosion du masque ciel de sigma/2 avant le flou normalisé — un pixel d'avant-plan lumineux ne contamine plus le fond estimé) et le poids de fusion ciel/avant-plan est **dilaté de sigma/2 avant le flou** : la correction s'applique à fond côté ciel jusqu'au bord du masque, la transition se fait côté avant-plan. Le liseré clair autour de l'avant-plan (balle) doit disparaître. **Choix du masque utilisé pour la PL** : « Peint (UI) » (défaut) ou **« ML ONNX (auto) »** — le masque ONNX est régénéré depuis la référence pour la correction PL (bords plus propres ; UI : combo « Masque PL », CLI : `--lp-mask {painted,ml}`, aussi pris en compte par `--lp-redo` et « Réappliquer la PL »). Nouveau helper `foreground_mask_from_linear()` dans `mlmask.py`.
- **v0.8.4** — **Checkbox des boîtes repliables distinctes** des cases standard : pastille **ronde 16 px** à bordure bleue, remplie quand cochée (QSS `#cbHeader::indicator`), contre le carré 14 px des autres. **Sauvegarde AVANT / APRÈS correction PL** : si la PL est activée, le stack linéaire est aussi écrit dans **`astrostack_result_avant_PL.tif`** avant correction. **Rejouer la correction PL sans re-stacker** : bouton **« Réappliquer la PL sur le dernier stack »** dans la box « Pollution lumineuse » (réapplique la correction avec les réglages actuels sur le dernier empilement terminé — la base non corrigée est conservée en mémoire via `meta_out`, donc on peut changer méthode/force et réappliquer sans cumul) ; en CLI : **`--lp-redo astrostack_result.tif`** → écrit `<base>_pl.tif` (+ aperçu, masque voisin `<base>_mask.png` utilisé). Fond « synthetic flat » estimé à **512 px** (au lieu de 256) : halo résiduel encore réduit.
- **v0.8.3** — **raccourcis `Ctrl+D` / `Ctrl+Maj+D` corrigés** : ils ajoutent bien des **darks** (`self._darks` / `list_darks`), et non plus des lights. **Masque ONNX (ML) dilaté de 2 px partout** (`cv2.dilate`, iterations=2, paramètre `grow=2` dans `sky_to_foreground_mask`). **Halo résiduel PL supprimé à la source** : le fond « synthetic flat » est désormais estimé **uniquement sur les pixels du ciel** (flou normalisé masqué `blur(img*sky)/blur(sky)`) — l'avant-plan lumineux (balle de foin) ne contamine plus le fond ; lissage du bord du masque renforcé, sigma auto passé de `min(h,l)/128` à **`min(h,l)/96`**.
- **v0.8.2** — base = `mainwindow-4.py` repris tel quel (panneau **340 px**, libellé « Ciel uniquement »). **Panneau gauche plus haut** (fenêtre 1120×860, min 880×620) et **agrandissement automatique de la fenêtre en hauteur quand on déplie une boîte** (repli = réduction) : les boîtes ne sont plus compressées (`widgetResizable(True)` + hauteurs mini des listes/log ; barre de défilement si l'écran est trop petit). **Anti-artefacts PL** : le masque de fusion ciel/avant-plan est désormais **lissé par flou gaussien** (sigma auto = min(h,l)/128) dans les deux méthodes — fini les halos au bord du masque peint (visibles sur l'aperçu v0.8.0/0.8.1). Paramètre `feather` exposé dans `correct_light_pollution`, `lp_feather` dans le pipeline et `--lp-feather` en CLI (0 = masque dur). Test mis à jour (vérifie l'absence de saut au bord du masque).
- **v0.8.1** — **correctif** : `UnboundLocalError: providers` avec la méthode **GraXpert** lors d'un *deuxième* lancement dans la même session (session ONNX déjà en cache : le bloc `if` était sauté et `providers` n'était plus assigné avant le `debug_cb`). `providers` est maintenant calculé **avant** le test de cache.
- **v0.8.0** — **correction de la pollution lumineuse (PL)** en post-empilement, en linéaire 16 bits avant l'export TIFF : nouvelle box repliée **« Pollution lumineuse »** (case d'activation, méthode, force en %, « ciel uniquement » via le masque avant-plan). Deux méthodes : **« Synthetic flat » (défaut)** — fond de ciel lissé estimé sur le masque ciel, soustrait canal par canal (additif), aucun modèle requis ; **« ML GraXpert (ONNX) »** — modèle officiel GraXpert background-extraction 1.0.1 (~218 Mo, licence **CC BY-NC-SA = usage non commercial**, téléchargé au premier usage dans `~/.astrostack/`, `pip install onnxruntime` requis). CLI : `--lp-correct`, `--lp-method {synthetic,graxpert}`, `--lp-strength`, `--no-lp-sky-only`. Nouveau test synthétique `test_lp_synthetic_flat`.
- **v0.7.15** — fichier utilisateur repris tel quel (`mainwindow-3.py` : raccourcis `Ctrl+D` / `Ctrl+Maj+D`, gomme sur `X`). **Taille du pinceau réglable au clavier et à la souris** : touches `+` / `-` (et `=` / `_` clavier US) par pas de 10 px, et **molette de la souris sur l'aperçu** (`eventFilter`, ±10 px) — synchronisé avec le slider « Pinceau ».
- **v0.7.14** — **correction utilisateur reprise telle quelle** (`mainwindow-2.py`) : panneau latéral **fixe à 320 px** (`widgetResizable(False)`, min=max=320), « Options » déplacé **sous** « Masque d'avant-plan », libellé « Gradient Ciel favorisé », bouton « ▶ Lancer ». **Raccourcis clavier** (`QShortcut`, Ctrl = ⌘ sur Mac) : `Ctrl+O` ajouter des images, `Ctrl+Maj+O` ajouter un dossier, `Ctrl+M` masque ML, `Ctrl+Entrée` lancer, `Échap` arrêter, `G` gomme, `I` inverser le masque, `F1` aide.
- **v0.7.13** — **panneau « left » compacté** (réponse « panneau trop grand + scroll ») : QSS `QGroupBox` (margin-top 12→4 px, padding réduit), `#CollapsibleBox` encore plus serrée, boutons padding 4→2 px, cases 14→12 px ; listes 48 px, log 64 px, barre de progression 14 px, espacements/marges des layouts et formulaires à 2 px.
- **v0.7.12** — **panneau latéral : retour à la largeur d'avant** (min 208 / max 288, splitter ~256 px) : un `setMinimumWidth(320)` forçait le panneau à ~288–320 px, comprimait l'aperçu à droite (barre de défilement qui réapparaissait) ; contenu compacté (listes 60 px, log 80 px, espacements réduits) et fenêtre par défaut 1120×760 pour éviter la barre de défilement verticale du panneau.
- **v0.7.11** — la section **« Options » est repliable** (CollapsibleBox comme Darks/Flats/Bias) ; **dégradé ONNX « ciel en haut » renforçable** : spin **« Facteur »** (0–3, défaut 0,5) à côté de la case à cocher, passé au masque via `top_boost` (0 = neutre, 1,0+ = fort).
- **v0.7.10** — **correction du sélecteur « Ajouter… »** : retour aux boîtes de dialogue **natives** (`QFileDialog.getOpenFileNames` / `getExistingDirectory`) pour « Ajouter… », « Dossier… » et le dossier de sortie — le sélecteur embarqué (QFileDialog non natif dans une fenêtre dédiée) pouvait s'ouvrir **sans liste de fichiers** ; sous macOS, le panneau natif affiche déjà les vignettes des images comme le Finder en mode icônes. Dernier dossier utilisé mémorisé (QSettings). **Bouton « Lancer »** : texte seul.
- **v0.7.9** — **référence changeable par clic** dans la liste des Images (ligne mise en gras, label « Référence : nom du fichier », défaut = image du milieu) ; **bouton « ⏹ Arrêter »** pour annuler l'empilement en cours (annulation coopérative `should_cancel`/`CancelledError` dans le pipeline) ; réglage « classe ciel » retiré de l'UI (env `ASTROSTACK_SKY_CLASS` toujours pris en compte) ; **masque ONNX : dégradé vertical « haut de l'image favorisé comme ciel »**, activable/désactivable par une case à cocher ; **sélecteur d'images avec volet d'aperçu à droite** (style Finder macOS, image contenue avec barre de déroulement, vignette RAW embarquée).
- **v0.7.8** — **bouton « Auto » + seuil supprimés de l'UI** (le CLI garde `--auto-mask`) ; **classe « ciel » ONNX réglable dans l'UI** : spin 0–149 à côté du bouton ML (défaut 2 = `sky` ADE20K, pré-rempli depuis `ASTROSTACK_SKY_CLASS` si défini), passée au masque et loguée (`ML : classe « ciel » = …`). La variable d'environnement reste utilisable.
- **v0.7.7** — **masque ML : retour au comportement v0.6.0** — l'inférence ONNX se fait sur la **référence à pleine résolution utile (plafond 1200 px)**, et plus sur l'aperçu réduit à la taille de la fenêtre (c'est ce qui produisait « ~100 % de ciel » depuis la v0.7.5). Le masque est ensuite redimensionné à l'aperçu affiché. Le repli luminance (ciel nocturne non reconnu) et les logs de proba sont conservés.
- **v0.7.6** — **masque ONNX réparé sur ciel nocturne** : le modèle SegFormer ADE20K ne reconnaît pas les cieux de nuit (proba « ciel » ≈ 0 → toute l'image partait en avant-plan). Ajout d'un **repli par luminance** (sombre = avant-plan) quand aucun ciel n'est détecté, correction du layout des logits (canaux en dernier), et **diagnostics logués** (proba ciel moyenne/max). **L'image de référence (base) est affichée dans l'UI** (label « Référence : … », image du milieu de la liste, `#N/total`) en plus du log. Tooltips explicites pour **Auto** (seuil de luminance : pixel plus sombre que la valeur = avant-plan) et le **chiffre à côté** (0–255, défaut 40).
- **v0.7.5** — **corrections GUI** : (1) crash de l'aperçu du sélecteur corrigé — `currentChanged` émet une `str` en PySide6/Qt6, plus une `QUrl` (géré des deux côtés) ; (2) la case **« Gomme » a de nouveau un état coché visible** (indicateur bleu en QSS sombre) avec log « Gomme activée / Pinceau » ; (3) **l'aperçu de la référence tient dans la fenêtre** : il est réduit pour entrer dans la zone visible (plafond 1200 px), la barre de déroulement gère le reste — le masque peint est re-redimensionné à la pleine résolution par le pipeline.
- **v0.7.4** — **correction du crash « Ajouter… »** : `QFileDialog.setPreviewWidget` a été supprimé en Qt6/PySide6 (`AttributeError`). Le volet d'aperçu des images est désormais ajouté **manuellement dans la grille du dialogue non natif** (rangée du bas, pleine largeur, hauteur fixe 170 px) : l'image y est **contenue avec barre de déroulement** ; vignette JPEG embarquée pour les RAW. Repli sur l'API officielle si présente, et `try/except` : **l'aperçu ne peut plus jamais bloquer la sélection de fichiers**.
- **v0.7.3** — **espace alloué du panneau élargi** (208–288 px, splitter 256/864) pour que les boîtes tiennent sans être coupées (défilement horizontal toujours désactivé). **Sélecteur de fichiers : panneau d'aperçu des images** comme dans les anciennes versions — volet fixe 212×180 **contenu dans la boîte de dialogue**, l'image y défile par **barre de déroulement** si elle dépasse ; les RAW (CR2/DNG/…) utilisent leur **vignette JPEG embarquée** (rawpy). **L'aperçu du résultat est ajusté à la taille de la fenêtre** (et le suit au redimensionnement). Dialogues toujours dimensionnés 720×460 et sans réouverture après ajout.
- **v0.7.2** — **panneau compact qui tient dans sa largeur** : 172–236 px, défilement horizontal désactivé, contenu resserré (listes 72 px, textes de boutons/combos raccourcis, cases ECC/TPS en pleine largeur, ligne d'outils masque sur 2 rangées) — plus rien n'est coupé. **Boîtes repliables : retour au look « cadre + case à cocher »** d'avant, mais avec une `QCheckBox` d'en-tête réellement cliquable (le `QGroupBox` checkable stylé en QSS ne répondait plus au clic). **Boîtes de dialogue de fichiers dimensionnées** (720×460, non natives) : elles tiennent dans l'espace alloué et ne se rouvrent pas après ajout.
- **v0.7.1** — **panneau encore réduit** (168–248 px au lieu de 230–320, listes 84 px) et **sections repliables réparées** : le `QGroupBox` checkable stylé en QSS ne répondait plus au clic (bug Qt connu) — l'en-tête des sections Images/Darks/Flats/Bias est maintenant un **bouton cliquable avec flèche ▾/▸**, le repli/dépli fonctionne partout. Par défaut seules Images et Darks sont ouvertes.
- **v0.7.0** — **interface revue et modernisée** : thème **sombre (Fusion + QSS)** identique sur Mac et Windows ; panneau de gauche **plus étroit (230–320 px)** ; sections **Images / Darks / Flats / Bias repliables individuellement** (case à cocher du titre, repli sur la hauteur) — **seules Images et Darks sont ouvertes par défaut** ; ajout de fichiers **sans réouverture de boîte de dialogue** : boutons séparés « Ajouter… » (fichiers) et « Dossier… » (dossier entier) ; chaque image a une **croix ✕** pour la retirer de sa liste ; l'**aperçu de la référence se charge automatiquement** dès l'ajout de lights ; la **référence est par défaut l'image du milieu de la liste** (`ref_index = N//2`, aussi `--ref-index N` en CLI).
- **v0.6.0** — **bords et avant-plan corrigés** : (1) chaque frame recalée est pondérée par son **masque de validité** (`warp_to_ref_valid`) dans l'empilement — les zéros de bordure du warp n'étaient pas distingués des vrais pixels noirs et produisaient une **bande sombre sur les bords** ; là où aucune frame ne couvre, on retombe sur la référence. (2) le masque d'avant-plan est **dilaté d'une marge de sécurité** (~0,5 %, min 3 px) avant le fondu, pour que le bord du masque tombe dans le ciel — sinon les pixels d'avant-plan au bord du masque étaient warpés, ce qui donnait des **arbres/sol fantômes**.
- **v0.5.3** — **bug fondamental d'alignement corrigé** : `warpAffine/warpPerspective` avec `WARP_INVERSE_MAP` calcule `dst(x) = src(M·x)` (M = échantillonnage dst→src), et *sans* le flag M est la carte forward src→dst. Le code retournait une carte forward en l'utilisant avec le flag inverse → double inversion, l'alignement était appliqué à l'envers (source des traînées persistantes, y compris dans les coins). Tout passe en convention forward + flags par défaut : `phaseCorrelate(a,b)=d ⟺ b(x)=a(x−d)` → translation **−d** ; `H_s = Hh @ M_t` ; `src_frame = inv(M_t)·src` ; rms réellement < 1 px ; ECC (`inv(G)`) ne s'ajoute au log que s'il a convergé (cc non NaN). Test : MSE ciel aligné doit chuter d'un facteur > 2, MSE bords aussi, et TPS > homographie en cas de distorsion.
- **v0.5.2** — correction du test et de la lecture TIFF : PIL/imageio ne gère pas le **PNG RGB 16 bits** (`TypeError: Cannot handle this data type: (1,1,3), <u2`) → le test écrit ses images synthétiques en **TIFF 16 bits via tifffile**, et `load_std_linear` lit les `.tif/.tiff` directement avec tifffile (les PNG 8 bits restent lus via imageio/PIL, idem pour les aperçus de sortie).
- **v0.5.1** — correction du test synthétique (`make_scene` : patch d'étoile non rogné quand l'étoile est proche du bord droit/bas → crash `ValueError: operands could not be broadcast`). Aucun changement du cœur.
- **v0.5.0** — **traînées dans les COINS corrigées** : la distorsion de l'objectif fait que la vraie transformation entre poses n'est pas une homographie (D∘R∘D⁻¹), l'erreur résiduelle croissant vers les coins. Ajout d'un **warp local par thin-plate spline (TPS)** : les appariements d'étoiles dont le résidu sous l'homographie est faible (répartis sur tout le champ, coins compris) ajustent une TPS dont la carte inverse sert de `cv2.remap` — l'homographie+ECC reste le repli si pas assez de points. Log : `méthode homographie+ecc+tps, N pts TPS`. Case « **Warp local (TPS)** » dans l'UI, `--no-local` en CLI. Test synthétique ajouté avec distorsion radiale : le TPS doit battre l'homographie sur les bords.
- **v0.4.0** — **traînées sur les bords corrigées** : l'alignement passe de la similitude à l'**homographie RANSAC** (modélise la perspective ; l'erreur de rotation croissait avec la distance au centre), calculée à échelle réduite (1600 px max) puis remise à pleine résolution, et affinée par **ECC photométrique** (`findTransformECC`, masque = ciel) — case « Affiner l'alignement sur les bords (ECC) » dans l'UI, `--no-ecc` en CLI, log `méthode homographie+ecc, rms < 1 px`. **Masque ML dé-bloqué** : probabilité softmax du canal ciel, upscalée en INTER_CUBIC + flou gaussien + morphologie 3x3 (fini les carrés de 512 px), toujours éditable au pinceau. Le masque auto par seuil est lissé pareil.
- **v0.3.0** — **dark/bias/flat soustraits sur le Bayer brut avant dématriçage** quand lights et calibration sont des RAW (gestion du black level par canal Bayer) ; **masque ML ONNX** (SegFormer-B0 ADE20K, téléchargement auto Hugging Face, cache `~/.astrostack/`), éditable au pinceau comme SharpStacker — bouton « Masque ML (ONNX) » + `--auto-mask ml` en CLI (extra `pip install .[ml]`) ; personnalisable via `ASTROSTACK_ONNX_MODEL` / `ASTROSTACK_ONNX_URL` / `ASTROSTACK_SKY_CLASS`.
- **v0.2.0** — alignement des étoiles **coarse-to-fine** : translation grossière par corrélation de phase fenêtrée (gère les grandes dérives), puis appariement d'étoiles (voisin réciproque + test de ratio de Lowe) et transformation similitude par RANSAC ; le log indique pour chaque image la méthode, le nombre d'inliers et le rms (px). Interface **plus compacte et repliable** : panneau de contrôle masquable (bouton « Panneau »), contenu scrollable, fenêtre redimensionnable avec géométrie mémorisée, aperçu du masque limité à 1200 px.
- **v0.1.1** — correction du crash `ValueError: operands could not be broadcast` quand les darks/flats/bias contenaient des images de tailles différentes (elles sont maintenant redimensionnées sur la référence) ; la liste des lights n'est plus triée (la 1ère image sélectionnée reste la référence sur laquelle on peint le masque) ; messages d'erreur nommés par fichier.
- **v0.1.0** — première version : GUI Qt + CLI, RAW via rawpy, alignement des étoiles, masque d'avant-plan peint au pinceau, darks/flats/bias, sigma clipping, export TIFF 16 bits.

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
  stacker.py     empilement : ciel recalé + sigma clipping, avant-plan fixe, raccord fondu
  pipeline.py    run_pipeline() : orchestration complète, export TIFF 16 bits + aperçus
  mlmask.py      masque ML ONNX (SegFormer ADE20K, sky=classe 2), téléchargeable, éditable au pinceau
  maskwidget.py  éditeur de masque au pinceau (Qt)
  mainwindow.py  fenêtre principale (Qt)
  cli.py         ligne de commande (--auto-mask heuristic|ml)
  imaging.py     tonemapping pour les aperçus
tests/test_synthetic.py
```
