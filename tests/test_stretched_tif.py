import os

import numpy as np
import tifffile

from astrostack.imaging import stretch_u16, tonemap_u16
from astrostack.pipeline import _save_result


def _linear_frame(h=40, w=60):
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 300, (h, w, 3)).astype(np.float32)
    img[:5, :5] = 60000.0  # quelques points très lumineux : les percentiles doivent les écrêter
    return img


def test_stretch_u16_is_16bit_and_brighter_than_linear():
    img = _linear_frame()
    out = stretch_u16(img)
    assert out.dtype == np.uint16 and out.shape == img.shape
    assert np.median(out) > np.median(img) * 10  # l'étirement et le gamma remontent un fond sombre
    # même forme que tonemap_u16 (à la quantification 8 bits près)
    np.testing.assert_allclose(out / 257.0, tonemap_u16(img).astype(np.float64), atol=1.5)


def test_save_result_writes_stretched_tif_only_when_asked(tmp_path):
    img = _linear_frame()
    fg = np.zeros(img.shape[:2], np.float32)
    out = str(tmp_path / "res.tif")

    paths = _save_result(out, img, fg)
    assert paths["stretched"] is None
    assert not os.path.exists(str(tmp_path / "res_etire.tif"))

    paths = _save_result(out, img, fg, stretched=True)
    assert paths["stretched"] == str(tmp_path / "res_etire.tif")
    linear = tifffile.imread(out)
    stretched = tifffile.imread(paths["stretched"])
    assert linear.dtype == np.uint16 and stretched.dtype == np.uint16
    assert stretched.mean() > linear.mean()  # le TIFF linéaire reste intact
