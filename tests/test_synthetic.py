"""Test de bout en bout sur des images synthétiques (pas besoin de RAW ni de Qt) :

  python tests/test_synthetic.py

Génère une scène (ciel étoilé + avant-plan), plusieurs poses décalées/tournées +
bruit + offset de dark, puis vérifie :
  1. l'alignement retrouve bien la transformation (MSE ciel < sans alignement) ;
  2. le repli par corrélation de phase aussi ;
  3. le pipeline complet : soustraction du dark, réduction du bruit dans le ciel,
     avant-plan resté fixe, export TIFF 16 bits.
"""
from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import tifffile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astrostack.pipeline import run_pipeline
from astrostack.stars import estimate_alignment, luminance, phase_translation, warp_to_ref


def test_lp_synthetic_flat():
    """La correction PL « synthetic flat » doit aplatir un fort gradient sur le ciel
    en laissant l'avant-plan (hors masque) inchangé."""
    from astrostack.lpcorrect import correct_light_pollution

    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = (3000.0 + 8000.0 * (xx / W) + 4000.0 * (yy / H)).astype(np.float32)  # fort dégradé PL
    img = np.repeat(grad[..., None], 3, axis=2) + rng.normal(0, 50, (H, W, 3)).astype(np.float32)
    img = np.clip(img, 0, 65535)
    sky = np.zeros((H, W), np.float32)
    sky[:FG_TOP] = 1.0  # ciel = haut de l'image, avant-plan = bas

    out = correct_light_pollution(img, method="synthetic", strength=1.0, sky=sky, sky_only=True)

    g_in = float(img[:FG_TOP, -1].mean() - img[:FG_TOP, 0].mean())
    g_out = float(out[:FG_TOP, -1].mean() - out[:FG_TOP, 0].mean())
    print(f"gradient PL : avant={g_in:.1f} après={g_out:.1f}")
    assert abs(g_out) < 0.2 * abs(g_in), "le gradient n'est pas assez aplati"
    # avant-plan (hors masque, loin du bord fondu) inchangé
    sigma = max(2.0, min(H, W) / 96.0)
    margin = int(6 * sigma) + 2
    assert np.allclose(out[FG_TOP + margin:], img[FG_TOP + margin:]), "l'avant-plan a été modifié !"
    # v0.8.2 : transition lissée au bord du masque -> pas de saut de correction (anti-halo)
    corr = np.abs(out - img).mean(axis=2)
    deep = float(corr[: FG_TOP // 2].mean())
    edge_jump = float(abs(corr[FG_TOP - 1].mean() - corr[FG_TOP].mean()))
    print(f"correction ciel={deep:.1f} saut au bord du masque={edge_jump:.1f}")
    assert edge_jump < 0.5 * deep, "halo au bord du masque (transition non lissée)"
    print("test_lp_synthetic_flat OK")

W, H = 900, 650
FG_TOP = int(H * 0.75)


def make_scene(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 900.0, np.float32)  # fond de ciel
    for _ in range(400):
        x, y = rng.uniform(0, W), rng.uniform(0, FG_TOP)
        amp = rng.uniform(4000, 60000)
        s = rng.uniform(0.7, 1.6)
        x0, y0 = int(x), int(y)
        r = 4
        ys0, ys1 = max(0, y0 - r), min(H, y0 + r + 1)
        xs0, xs1 = max(0, x0 - r), min(W, x0 + r + 1)
        yy, xx = np.mgrid[ys0:ys1, xs0:xs1]
        patch = amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
        img[ys0:ys1, xs0:xs1] += patch[..., None]
    # avant-plan : sol sombre texturé
    fg = np.full((H - FG_TOP, W, 3), 18000.0, np.float32)
    fg += rng.normal(0, 800, fg.shape)
    img[FG_TOP:] = fg
    return np.clip(img, 0, 65535)


def make_frame(ref, tx, ty, rot_deg, rng):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    # frame(x) = ref(M·x) avec WARP_INVERSE_MAP, donc la carte forward frame -> ref vaut M
    frame = cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    frame = frame + rng.normal(0, 250, frame.shape) + 600.0  # bruit + offset dark
    return np.clip(frame, 0, 65535), M


def main():
    tmp = tempfile.mkdtemp(prefix="astrostack_test_")
    ref = make_scene()
    rng = np.random.default_rng(42)

    lights, transforms = [], []
    for i in range(6):
        tx, ty = rng.uniform(-14, 14), rng.uniform(-9, 9)
        rot = rng.uniform(-1.2, 1.2)
        frame, M = make_frame(ref, tx, ty, rot, rng)
        transforms.append(M)
        p = os.path.join(tmp, f"light_{i:02d}.tif")  # TIFF 16 bits (PIL ne gère pas le PNG RGB uint16)
        tifffile.imwrite(p, frame.astype(np.uint16))
        lights.append(p)

    dark = np.clip(np.full((H, W, 3), 600.0, np.float32) + rng.normal(0, 30, (H, W, 3)), 0, 65535)
    dark_path = os.path.join(tmp, "dark.tif")
    tifffile.imwrite(dark_path, dark.astype(np.uint16))

    mask = np.zeros((H, W), np.uint8)
    mask[FG_TOP:] = 255

    # --- 1. alignement v0.5.0 (phase + étoiles + homographie RANSAC + ECC + TPS local) ---
    ref_gray = luminance(ref)
    sky = (1.0 - mask.astype(np.float32) / 255.0)
    sky_bool = mask == 0
    frame0 = cv2.warpAffine(ref, transforms[0], (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    M_est, info = estimate_alignment(ref_gray, luminance(frame0), sky, refine_ecc=True)
    print("info alignement (ECC on) :", info)
    ok_shape = (isinstance(M_est, tuple) and M_est[0] == "remap") or getattr(M_est, "shape", None) == (3, 3)
    assert ok_shape, "retour attendu : homographie 3x3 ou tuple ('remap', map_x, map_y)"
    aligned = warp_to_ref(frame0, M_est, (W, H))
    mse_aligned = float(np.mean((aligned[sky_bool] - ref[sky_bool]) ** 2))
    mse_raw = float(np.mean((frame0[sky_bool] - ref[sky_bool]) ** 2))
    print(f"MSE ciel aligné={mse_aligned:.1f} vs non aligné={mse_raw:.1f}")
    assert mse_aligned < mse_raw * 0.5, "l'alignement ne améliore pas le ciel"

    # variante sans ECC (doit quand même fonctionner, homographie seule)
    M_noecc, info2 = estimate_alignment(ref_gray, luminance(frame0), sky, refine_ecc=False)
    aligned2 = warp_to_ref(frame0, M_noecc, (W, H))
    mse_noecc = float(np.mean((aligned2[sky_bool] - ref[sky_bool]) ** 2))
    print(f"MSE ciel aligné sans ECC={mse_noecc:.1f} (méthode {info2['method']})")
    assert mse_noecc < mse_raw * 0.5, "l'alignement sans ECC ne améliore pas le ciel"

    # les bords (coins) doivent être bien recalés aussi, pas seulement le centre
    edge = np.zeros((H, W), bool)
    edge[:60, :] = edge[-60:, :] = True
    edge[:, :60] = edge[:, -60:] = True
    edge &= sky_bool
    mse_edge = float(np.mean((aligned[edge] - ref[edge]) ** 2))
    print(f"MSE bords aligné={mse_edge:.1f}")
    assert mse_edge < mse_raw, "les bords ne sont pas mieux recalés que le non-aligné"

    # --- 1b. distorsion de l'objectif : le warp local TPS doit battre l'homographie seule ---
    def distort(img, k):
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rn2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (max(h, w) / 2.0) ** 2
        mx = cx + (1.0 + k * rn2) * (xx - cx)
        my = cy + (1.0 + k * rn2) * (yy - cy)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    frame_d = np.clip(distort(ref, -0.12) + rng.normal(0, 250, (H, W, 3)) + 600.0, 0, 65535)
    g_d = luminance(frame_d)
    M_loc, info_loc = estimate_alignment(ref_gray, g_d, sky, refine_local=True, min_local=30)
    M_hom, info_hom = estimate_alignment(ref_gray, g_d, sky, refine_local=False)
    print("distorsion — TPS:", info_loc["method"], "| homographie seule:", info_hom["method"])
    assert "tps" in info_loc["method"], "le warp local TPS aurait dû s'activer"
    al_loc = warp_to_ref(frame_d, M_loc, (W, H))
    al_hom = warp_to_ref(frame_d, M_hom, (W, H))
    mse_edge_loc = float(np.mean((al_loc[edge] - ref[edge]) ** 2))
    mse_edge_hom = float(np.mean((al_hom[edge] - ref[edge]) ** 2))
    print(f"MSE bords avec distorsion : TPS={mse_edge_loc:.1f} vs homographie={mse_edge_hom:.1f}")
    assert mse_edge_loc < mse_edge_hom, "le TPS n'améliore pas les bords en cas de distorsion"

    # --- 2. translation seule par corrélation de phase ---
    M_t, resp = phase_translation(ref_gray, luminance(frame0), sky)
    aligned_fb = warp_to_ref(frame0, M_t, (W, H))
    mse_fb = float(np.mean((aligned_fb[sky_bool] - ref[sky_bool]) ** 2))
    print(f"MSE corrélation de phase seule={mse_fb:.1f} (resp={resp:.3f})")
    assert mse_fb < mse_raw * 0.5, "la corrélation de phase seule est incorrecte (signe ?)"

    # --- 3. pipeline complet ---
    out = os.path.join(tmp, "result.tif")
    result = run_pipeline(
        lights=lights,
        darks=[dark_path],
        mask=mask,
        scale=1.0,
        mode="sigma",
        kappa=2.5,
        out_path=out,
        progress_cb=lambda s, i, n, m: None,
    )
    assert result.shape == (H, W, 3)
    assert os.path.exists(out) and os.path.exists(out.replace(".tif", "_preview.png"))

    # dark correctement soustrait + bruit réduit dans le ciel
    single = np.clip(tifffile.imread(lights[1]).astype(np.float32) - dark, 0, 65535)
    std_single = float(np.std(single[sky_bool]))
    std_stack = float(np.std(result[sky_bool] - ref[sky_bool]))
    print(f"bruit ciel : pose seule std={std_single:.1f}, stacké écart à la vérité std={std_stack:.1f}")
    assert std_stack < std_single * 0.8, "le stacking ne réduit pas le bruit"

    # avant-plan fixe : proche de la référence (non flou)
    fg_bool = mask > 0
    fg_err = float(np.mean(np.abs(result[fg_bool] - ref[fg_bool])))
    print(f"erreur avant-plan vs référence : {fg_err:.1f}")
    assert fg_err < 400, "l'avant-plan n'est pas resté fixe/net"

    print("OK — tous les tests passent. Dossier temporaire :", tmp)


if __name__ == "__main__":
    test_lp_synthetic_flat()
    main()
