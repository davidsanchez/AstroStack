"""Interface en ligne de commande (sans GUI) :

    python -m astrostack.cli --lights-dir ./lights --darks-dir ./darks \
        --mask masque.png --out resultat.tif --mode sigma --kappa 2.5
"""
from __future__ import annotations

import argparse

import numpy as np

from .pipeline import run_pipeline
from .rawio import list_image_files, load_image_linear


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="astrostack-cli", description="Empilement de paysages de nuit (CLI)")
    p.add_argument("--lights", nargs="+", default=[], help="fichiers lights (RAW ou TIFF/PNG) — la 1ère = référence")
    p.add_argument("--lights-dir", help="dossier contenant les lights (triés par nom ; le 1er = référence)")
    p.add_argument("--darks", nargs="*", default=[])
    p.add_argument("--darks-dir")
    p.add_argument("--flats", nargs="*", default=[])
    p.add_argument("--flats-dir")
    p.add_argument("--bias", nargs="*", default=[])
    p.add_argument("--bias-dir")
    p.add_argument("--mask", help="PNG 8 bits : 255 = avant-plan (fixe), 0 = ciel (aligné)")
    p.add_argument(
        "--auto-mask",
        choices=["none", "heuristic", "ml"],
        default="none",
        help="génère le masque depuis la 1ère light : 'heuristic' (seuil) ou 'ml' (ONNX, nécessite onnxruntime)",
    )
    p.add_argument(
        "--no-mask-guided",
        action="store_true",
        help="v0.9.0 : désactive le filtre guidé et le seuil local du masque ML (retour au comportement v0.8.x)",
    )
    p.add_argument(
        "--no-mask-lum",
        action="store_true",
        help="v0.8.11 : désactive l'affinage du masque ML par luminance (défaut : ACTIF avec "
        "--auto-mask ml — avant-plan sombre forcé en avant-plan, grandes zones très lumineuses "
        "(lueur d'horizon) basculées en ciel)",
    )
    p.add_argument(
        "--mask-lum-thr",
        type=int,
        default=0,
        help="v0.8.11 : seuil de luminance 0..255 pour l'affinage du masque ML (0 = auto, seuil d'Otsu)",
    )
    p.add_argument(
        "--mask-edge-band",
        type=int,
        default=40,
        help="v0.8.14 : largeur en px de la bande autour du bord du masque ONNX où le bord est recalé "
        "sur la luminance pleine résolution (détails fins : branches) ; 0 = désactivé (défaut 40)",
    )
    p.add_argument(
        "--stack-grow",
        type=int,
        default=5,
        help="v1.1.2 : marge d'empilement en px (défaut 5) — élargit l'avant-plan pour le seul "
        "empilement, contre les fantômes autour des arbres ; ni affichée, ni utilisée par la PL",
    )
    p.add_argument(
        "--mask-max-side",
        type=int,
        default=2000,
        help="v0.8.14 : grand côté (px) de l'image de travail pour le calcul du masque ML "
        "(défaut 2000 ; 1200 avant) — plus grand = détails plus fins mais plus lent",
    )
    p.add_argument("--out", default="astrostack_result.tif")
    p.add_argument("--mode", choices=["sigma", "robust", "mean"], default="sigma",
                   help="sigma (défaut) ; robust = rejet centré sur la médiane (expérimental) ; mean = moyenne simple")
    p.add_argument("--kappa", type=float, default=2.5)
    p.add_argument(
        "--no-ecc",
        action="store_true",
        help="désactive l'affinage photométrique ECC de l'alignement (par défaut actif : "
        "corrige les traînées d'étoiles résiduelles sur les bords)",
    )
    p.add_argument(
        "--no-local",
        action="store_true",
        help="désactive le warp local par thin-plate spline (TPS) — par défaut actif, il "
        "corrige la distorsion de l'objectif et les traînées dans les COINS de l'image",
    )
    p.add_argument("--scale", type=float, default=1.0, help="1.0 = pleine résolution, 0.5 = moitié…")
    p.add_argument("--fast", action="store_true",
                   help="mode Rapide (peut légèrement modifier le résultat ; défaut = Précis)")
    p.add_argument("--load-threads", type=int, default=2,
                   help="RAW lus/dématricés en parallèle (lights et masters ; défaut 2)")
    p.add_argument("--demosaic", choices=["auto", "ahd", "ppg", "linear"], default="auto",
                   help="algorithme de dématriçage RAW (auto = AHD en Précis, PPG en Rapide)")
    p.add_argument(
        "--ref-index",
        type=int,
        default=0,
        help="index de la light de référence dans la liste (défaut 0 = la 1ère ; ex. milieu = N/2)",
    )
    p.add_argument(
        "--lp-correct",
        action="store_true",
        help="correction de la pollution lumineuse : fond de ciel estimé sur le ciel détecté par le "
        "masque automatique (ONNX), soustrait en linéaire après l'empilement",
    )
    p.add_argument("--lp-strength", type=float, default=1.0, help="force de la correction PL, 0..1 (défaut 1.0)")
    p.add_argument(
        "--lp-feather",
        type=float,
        default=None,
        help="sigma en px du lissage de la transition au bord du masque en correction PL "
        "(défaut auto = min(h,l)/96, anti-halos ; 0 = masque dur)",
    )
    p.add_argument(
        "--lp-redo",
        metavar="TIFF",
        help="v0.8.4 : réapplique UNIQUEMENT la correction PL sur un TIFF déjà empilé "
        "(ex. astrostack_result.tif), sans re-stacker : écrit <base>_pl.tif + aperçu. "
        "Le ciel est détecté automatiquement sur ce TIFF (masque automatique).",
    )
    args = p.parse_args(argv)

    if args.lp_redo:  # correction PL seule, sans empilement
        import os

        import imageio.v3 as iio
        import tifffile

        from .lpcorrect import auto_sky_weight, correct_light_pollution
        from .pipeline import _save_result

        src = args.lp_redo
        img = tifffile.imread(src).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        base, _ = os.path.splitext(src)
        sky = auto_sky_weight(img, debug_cb=lambda m: print("[ML]", m, flush=True))
        out = correct_light_pollution(
            img, sky, strength=args.lp_strength, feather=args.lp_feather,
            debug_cb=lambda m: print("[PL]", m, flush=True),
        )
        iio.imwrite(base + "_pl_maskPL.png", (np.clip(sky, 0.0, 1.0) * 255.0).round().astype(np.uint8))
        print(f"Masque PL écrit : {base}_pl_maskPL.png — ciel = {float((sky > 0.5).mean()):.0%}", flush=True)
        paths = _save_result(base + "_pl.tif", out, np.zeros(img.shape[:2], np.float32))
        print("Terminé :", paths, flush=True)
        return

    lights = list(args.lights) + (list_image_files(args.lights_dir) if args.lights_dir else [])
    darks = list(args.darks) + (list_image_files(args.darks_dir) if args.darks_dir else [])
    flats = list(args.flats) + (list_image_files(args.flats_dir) if args.flats_dir else [])
    biases = list(args.bias) + (list_image_files(args.bias_dir) if args.bias_dir else [])
    if not lights:
        p.error("aucun light fourni (--lights ou --lights-dir)")

    mask = None
    if args.mask:
        import imageio.v3 as iio

        m = iio.imread(args.mask)
        if m.ndim == 3:
            m = m[..., 0]
        mask = m.astype(np.uint8)
    elif args.auto_mask != "none":
        import cv2

        img = load_image_linear(lights[0]).astype(np.float32)
        h, w = img.shape[:2]
        s = min(1.0, args.mask_max_side / max(h, w))  # résolution de travail réglable (défaut 2000 px)
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        from .imaging import tonemap_u16

        u8 = tonemap_u16(img)
        if args.auto_mask == "ml":
            from .mlmask import sky_to_foreground_mask

            mask = sky_to_foreground_mask(  # affinage luminance ; grow=0 ; bord luminance + bande
                u8, grow=0, lum_refine=not args.no_mask_lum, lum_thr=(args.mask_lum_thr or None),
                lum_edge=args.mask_edge_band > 0, edge_band=args.mask_edge_band,
                guided=not args.no_mask_guided,  # filtre guidé pleine résolution (détails fins)
            )
        else:
            lum = (0.2126 * u8[..., 0] + 0.7152 * u8[..., 1] + 0.0722 * u8[..., 2]).astype(np.uint8)
            mask = np.where(lum <= 40, 255, 0).astype(np.uint8)
        print(f"Masque auto ({args.auto_mask}) généré depuis {lights[0]}", flush=True)

    def cb(stage, i, n, msg):
        print(f"[étape {stage}] {i}/{n} {msg}", flush=True)

    run_pipeline(
        lights=lights,
        darks=darks or None,
        flats=flats or None,
        biases=biases or None,
        mask=mask,
        scale=args.scale,
        mode=args.mode,
        kappa=args.kappa,
        refine_ecc=not args.no_ecc,
        refine_local=not args.no_local,
        ref_index=args.ref_index,
        out_path=args.out,
        lp_correct=args.lp_correct,
        lp_strength=args.lp_strength,
        lp_feather=args.lp_feather,
        stack_grow=args.stack_grow,
        progress_cb=cb,
        fast=args.fast,
        load_threads=args.load_threads,
        demosaic=args.demosaic,
    )
    print("Terminé :", args.out)


if __name__ == "__main__":
    main()
