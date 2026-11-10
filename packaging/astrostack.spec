# -*- mode: python ; coding: utf-8 -*-
"""Build PyInstaller d'AstroStack (macOS : AstroStack.app ; Windows : dossier AstroStack/).

Lancer depuis la racine du dépôt :  pyinstaller --noconfirm packaging/astrostack.spec
Les icônes sont générées par packaging/make_icons.py (dossier packaging/build).
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (SPECPATH fourni par PyInstaller)
BUILD = os.path.join(SPECPATH, "build")  # noqa: F821
ICO = os.path.join(BUILD, "icon.ico")
ICNS = os.path.join(BUILD, "icon.icns")

with open(os.path.join(ROOT, "astrostack", "__init__.py"), encoding="utf-8") as fh:
    VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', fh.read()).group(1)

# onnxruntime est inclus : le bouton « Masque ML » fonctionne sans installation supplémentaire
a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "run.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AstroStack",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # application fenêtrée, pas de terminal
    icon=ICO if os.path.exists(ICO) else None,
)
coll = COLLECT(  # noqa: F821
    exe, a.binaries, a.datas, name="AstroStack", strip=False, upx=False,
)

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name="AstroStack.app",
        icon=ICNS if os.path.exists(ICNS) else None,
        bundle_identifier="fr.davidsanchez.astrostack",
        info_plist={
            "CFBundleName": "AstroStack",
            "CFBundleDisplayName": "AstroStack",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSHumanReadableCopyright": "AstroStack — licence du dépôt",
        },
    )
