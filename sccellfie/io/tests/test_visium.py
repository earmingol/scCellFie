import json

import numpy as np
import pytest
from anndata import AnnData

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon

from sccellfie.io import read_visium
from sccellfie.io import visium as visium_mod


@pytest.fixture
def fake_segmented_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "hd_bundle"
    bundle.mkdir()
    (bundle / "spatial").mkdir()

    barcodes = ["cellid_000000001-1", "cellid_000000002-1"]

    def _fake_read_10x_h5(path, genome=None):
        a = AnnData(X=np.zeros((2, 1), dtype=np.float32))
        a.obs_names = barcodes
        a.var_names = ["g1"]
        return a

    monkeypatch.setattr(visium_mod, "read_10x_h5", _fake_read_10x_h5)

    class _FakeAttrs:
        def __init__(self):
            self._d = {"library_ids": [b"libA"]}

        def __iter__(self):
            return iter(self._d)

        def __getitem__(self, k):
            return self._d[k]

        def keys(self):
            return self._d.keys()

        def items(self):
            return self._d.items()

    class _FakeFile:
        def __init__(self, *a, **kw):
            self.attrs = _FakeAttrs()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(visium_mod, "File", _FakeFile)

    (bundle / "filtered_feature_bc_matrix.h5").write_bytes(b"")

    scalefactors = {
        "tissue_hires_scalef": 1.0,
        "tissue_lowres_scalef": 0.5,
        "spot_diameter_fullres": 1.0,
        "microns_per_pixel": 0.5,
    }
    (bundle / "spatial" / "scalefactors_json.json").write_text(json.dumps(scalefactors))

    cell_polys = gpd.GeoDataFrame(
        {
            "cell_id": [1, 2],
            "geometry": [
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                Polygon([(20, 0), (28, 0), (28, 8), (20, 8)]),
            ],
        },
        geometry="geometry",
    )
    cell_polys.to_file(bundle / "cell_segmentations.geojson", driver="GeoJSON")

    return bundle, barcodes


def test_read_visium_segmented(fake_segmented_bundle):
    bundle, barcodes = fake_segmented_bundle
    adata = read_visium(bundle, is_hd=True, hd_layout="detect", load_images=False)

    assert adata.n_obs == 2
    assert "spatial" in adata.obsm
    assert "X_spatial" in adata.obsm
    np.testing.assert_array_equal(adata.obsm["spatial"], adata.obsm["X_spatial"])
    assert "cell_area" in adata.obs.columns
    np.testing.assert_allclose(
        adata.obs["cell_area"].values,
        np.array([100.0, 64.0]) * (0.5 ** 2),
    )

    sf = adata.uns["spatial"]["libA"]["scalefactors"]
    assert "bin_size_um" in sf
    assert sf["bin_size_um"] > 0
    assert adata.uns["spatial"]["libA"]["metadata"]["visium_hd"] is True
    assert adata.uns["spatial"]["libA"]["metadata"]["hd_layout"] == "segmented"


def test_read_visium_segmented_with_nucleus(fake_segmented_bundle):
    bundle, _ = fake_segmented_bundle
    nucleus_polys = gpd.GeoDataFrame(
        {
            "cell_id": [1, 2],
            "geometry": [
                Polygon([(2, 2), (8, 2), (8, 8), (2, 8)]),
                Polygon([(22, 2), (26, 2), (26, 6), (22, 6)]),
            ],
        },
        geometry="geometry",
    )
    nucleus_polys.to_file(bundle / "nucleus_segmentations.geojson", driver="GeoJSON")

    adata = read_visium(bundle, is_hd=True, load_images=False)
    assert "nucleus_area" in adata.obs.columns
    np.testing.assert_allclose(
        adata.obs["nucleus_area"].values,
        np.array([36.0, 16.0]) * (0.5 ** 2),
    )


def test_read_visium_standard_delegates(monkeypatch, tmp_path):
    captured = {}

    def _fake_sc_read_visium(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        a = AnnData(X=np.zeros((1, 1), dtype=np.float32))
        a.obs_names = ["bc"]
        a.var_names = ["g"]
        a.obsm["spatial"] = np.array([[1.0, 2.0]])
        return a

    monkeypatch.setattr(visium_mod.sc, "read_visium", _fake_sc_read_visium)

    out = read_visium(tmp_path, is_hd=False, load_images=False)
    assert captured["path"] == tmp_path
    assert "X_spatial" in out.obsm
    np.testing.assert_array_equal(out.obsm["X_spatial"], out.obsm["spatial"])


def test_read_visium_segmented_missing_geojson(tmp_path, monkeypatch):
    bundle = tmp_path / "broken"
    bundle.mkdir()
    (bundle / "spatial").mkdir()
    (bundle / "filtered_feature_bc_matrix.h5").write_bytes(b"")
    (bundle / "spatial" / "scalefactors_json.json").write_text("{}")

    # Force segmented path even though geojson is missing
    with pytest.raises((OSError, FileNotFoundError)):
        read_visium(bundle, is_hd=True, hd_layout="segmented", load_images=False)
