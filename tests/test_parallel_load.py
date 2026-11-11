"""Lectures parallèles : même résultat qu'en séquentiel, ordre préservé, choix de dématriçage validé."""
import numpy as np
import tifffile
import pytest

from astrostack.pipeline import run_pipeline
from astrostack.rawio import _map_ordered, build_master


def test_map_ordered_keeps_order_for_any_worker_count():
    items = list(range(23))
    for workers in (1, 2, 3, 5):
        assert list(_map_ordered(lambda x: x * x, items, workers)) == [x * x for x in items]


def test_master_is_identical_with_parallel_reads(tmp_path):
    rng = np.random.default_rng(1)
    paths = []
    for k in range(5):
        p = tmp_path / f"dark{k}.tif"
        tifffile.imwrite(str(p), rng.integers(0, 4000, (40, 60, 3), dtype=np.uint16))
        paths.append(str(p))
    seq = build_master(paths, "dark", workers=1)
    par = build_master(paths, "dark", workers=3)
    assert np.array_equal(seq, par)


def test_unknown_demosaic_is_rejected():
    with pytest.raises(ValueError):
        run_pipeline(lights=["x.tif"], demosaic="bogus")


def test_load_threads_must_be_positive():
    with pytest.raises(ValueError):
        run_pipeline(lights=["x.tif"], load_threads=0)


class _FakeRaw:
    def __init__(self, fail_dcb=False):
        self.calls = []
        self.fail_dcb = fail_dcb

    def postprocess(self, **kw):
        self.calls.append(kw)
        if self.fail_dcb and "dcb_enhance" in kw:
            raise TypeError("dcb_enhance inconnu")
        return np.zeros((2, 2, 3), np.uint16)


@pytest.fixture
def fake_rawpy(monkeypatch):
    import sys
    import types

    mod = types.SimpleNamespace(
        DemosaicAlgorithm=types.SimpleNamespace(AHD="AHD", PPG="PPG", LINEAR="LINEAR")
    )
    monkeypatch.setitem(sys.modules, "rawpy", mod)
    return mod


def test_demosaic_algorithm_is_passed_to_libraw(fake_rawpy):
    from astrostack.rawio import _postprocess

    raw = _FakeRaw()
    _postprocess(raw, {"output_bps": 16}, "ppg")
    assert raw.calls[0]["demosaic_algorithm"] == "PPG"
    raw = _FakeRaw()
    _postprocess(raw, {"output_bps": 16}, None)  # défaut LibRaw : aucun argument ajouté
    assert "demosaic_algorithm" not in raw.calls[0]


def test_demosaic_kept_when_dcb_option_unsupported(fake_rawpy):
    from astrostack.rawio import _postprocess

    raw = _FakeRaw(fail_dcb=True)
    _postprocess(raw, {"output_bps": 16}, "linear")
    assert len(raw.calls) == 2
    assert raw.calls[1]["demosaic_algorithm"] == "LINEAR"
    assert "dcb_enhance" not in raw.calls[1]
