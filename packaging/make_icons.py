"""Icône d'AstroStack.

Usage :
  python packaging/make_icons.py            -> génère packaging/build/icon.ico (+ icon.icns sur macOS)
  python packaging/make_icons.py --draw     -> redessine packaging/icon.png (1024 px, source versionnée)

Le dessin est procédural (ciel étoilé, crête de montagne, étoile lumineuse), sans image externe.
"""
from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PNG = os.path.join(HERE, "icon.png")
BUILD = os.path.join(HERE, "build")
SIZES = [16, 32, 48, 64, 128, 256, 512, 1024]


def draw(size: int = 1024) -> Image.Image:
    rnd = random.Random(7)
    s = size
    # fond : dégradé nuit (haut sombre -> bas bleu nuit)
    bg = Image.new("RGB", (s, s))
    px = bg.load()
    top, bottom = (6, 12, 28), (24, 52, 92)
    for y in range(s):
        t = y / (s - 1)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(s):
            px[x, y] = c
    img = bg.convert("RGBA")

    # étoiles
    stars = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(stars)
    for _ in range(260):
        x, y = rnd.uniform(0, s), rnd.uniform(0, s * 0.62)
        r = rnd.choice([1.5, 2, 2.5, 3]) * s / 512
        a = rnd.randint(140, 255)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 242, 255, a))
    img = Image.alpha_composite(img, stars)

    # étoile lumineuse avec halo
    cx, cy = s * 0.70, s * 0.27
    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([cx - s * 0.10, cy - s * 0.10, cx + s * 0.10, cy + s * 0.10], fill=(160, 205, 255, 200))
    glow = glow.filter(ImageFilter.GaussianBlur(s * 0.06))
    img = Image.alpha_composite(img, glow)
    sd = ImageDraw.Draw(img)
    sd.ellipse([cx - s * 0.018, cy - s * 0.018, cx + s * 0.018, cy + s * 0.018], fill=(255, 255, 255, 255))
    # pointes en croix
    for k in range(4):
        ang = math.radians(90 * k)
        L = s * 0.075
        sd.line([cx, cy, cx + L * math.cos(ang), cy - L * math.sin(ang)], fill=(255, 255, 255, 200), width=max(2, s // 170))

    # crêtes de montagne (plusieurs plans, du lointain au proche)
    def ridge(base: float, amp: float, color, seed: int, freq: float):
        r = random.Random(seed)
        phases = [r.uniform(0, 6.28) for _ in range(4)]
        pts = [(0, s)]
        for x in range(0, s + 1, max(2, s // 128)):
            y = base * s + amp * s * (
                0.5 * math.sin(freq * x / s * 6.28 + phases[0])
                + 0.3 * math.sin(freq * 2.3 * x / s * 6.28 + phases[1])
                + 0.2 * math.sin(freq * 4.7 * x / s * 6.28 + phases[2])
            )
            pts.append((x, y))
        pts.append((s, s))
        ImageDraw.Draw(img).polygon(pts, fill=color)

    ridge(0.70, 0.07, (18, 40, 66, 255), 3, 1.1)
    ridge(0.80, 0.05, (10, 26, 44, 255), 5, 0.8)
    ridge(0.90, 0.03, (4, 14, 26, 255), 9, 0.6)

    # masque arrondi (coins transparents, marge type macOS)
    margin = int(s * 0.06)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [margin, margin, s - margin - 1, s - margin - 1], radius=int(s * 0.19), fill=255)
    out = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out


def build_ico(src: Image.Image) -> str:
    path = os.path.join(BUILD, "icon.ico")
    src.resize((256, 256), Image.LANCZOS).save(path, sizes=[(n, n) for n in [16, 32, 48, 64, 128, 256]])
    return path


def build_icns(src: Image.Image) -> str | None:
    if sys.platform != "darwin" or not shutil.which("iconutil"):
        return None
    iconset = os.path.join(BUILD, "icon.iconset")
    shutil.rmtree(iconset, ignore_errors=True)
    os.makedirs(iconset)
    specs = {
        "icon_16x16.png": 16, "icon_16x16@2x.png": 32, "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128, "icon_128x128@2x.png": 256, "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512, "icon_512x512.png": 512, "icon_512x512@2x.png": 1024,
    }
    for name, n in specs.items():
        src.resize((n, n), Image.LANCZOS).save(os.path.join(iconset, name))
    out = os.path.join(BUILD, "icon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)
    shutil.rmtree(iconset)
    return out


def main() -> None:
    os.makedirs(BUILD, exist_ok=True)
    if "--draw" in sys.argv or not os.path.exists(SRC_PNG):
        draw(1024).save(SRC_PNG)
        print("écrit", SRC_PNG)
    src = Image.open(SRC_PNG).convert("RGBA")
    print("ico ->", build_ico(src))
    icns = build_icns(src)
    print("icns ->", icns or "ignoré (pas macOS)")


if __name__ == "__main__":
    main()
