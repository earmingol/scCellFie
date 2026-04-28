import gzip

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

from sccellfie.io import read_xenium


@pytest.fixture
def fake_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "slide1"
    bundle.mkdir()

    barcodes = ["bc1", "bc2", "bc3"]

    def _fake_read_10x_h5(path):
        # Return a minimal AnnData regardless of file contents
        a = AnnData(X=np.zeros((3, 2), dtype=np.float32))
        a.obs_names = barcodes
        a.var_names = ["gene1", "gene2"]
        return a

    monkeypatch.setattr(
        "sccellfie.io.xenium.sc.read_10x_h5", _fake_read_10x_h5
    )

    (bundle / "cell_feature_matrix.h5").write_bytes(b"")

    cells = pd.DataFrame(
        {
            "x_centroid": [10.0, 20.0, 30.0],
            "y_centroid": [1.0, 2.0, 3.0],
            "total_counts": [100, 200, 150],
        }
    )
    with gzip.open(bundle / "cells.csv.gz", "wt") as fh:
        cells.to_csv(fh, index=False)

    return bundle, barcodes


def test_read_xenium_cell_segmentation(fake_bundle):
    bundle, barcodes = fake_bundle
    adata = read_xenium(bundle, segmentation="cell", verbose=False)

    assert adata.n_obs == 3
    assert "X_spatial" in adata.obsm
    np.testing.assert_allclose(adata.obsm["X_spatial"][:, 0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(adata.obsm["X_spatial"][:, 1], [1.0, 2.0, 3.0])
    assert "x_centroid" not in adata.obs.columns
    assert "total_counts" in adata.obs.columns


def test_read_xenium_with_clusters(fake_bundle):
    bundle, barcodes = fake_bundle
    cluster_dir = bundle / "analysis" / "clustering" / "gene_expression_graphclust"
    cluster_dir.mkdir(parents=True)
    pd.DataFrame({"Barcode": barcodes, "Cluster": ["3", "1", "2"]}).to_csv(
        cluster_dir / "clusters.csv", index=False
    )

    adata = read_xenium(bundle, segmentation="cell", verbose=False)

    assert "cluster" in adata.obs.columns
    assert list(adata.obs["cluster"].cat.categories) == ["1", "2", "3"]


def test_read_xenium_skip_clusters(fake_bundle):
    bundle, _ = fake_bundle
    adata = read_xenium(bundle, segmentation="cell", cluster_file=False, verbose=False)
    assert "cluster" not in adata.obs.columns


def test_read_xenium_explicit_cluster_path_missing(fake_bundle):
    bundle, _ = fake_bundle
    with pytest.raises(FileNotFoundError):
        read_xenium(bundle, segmentation="cell", cluster_file=str(bundle / "nope.csv"), verbose=False)


def test_read_xenium_custom_spatial_key(fake_bundle):
    bundle, _ = fake_bundle
    adata = read_xenium(bundle, segmentation="cell", spatial_key="spatial", verbose=False)
    assert "spatial" in adata.obsm
    assert "X_spatial" not in adata.obsm


def test_read_xenium_invalid_segmentation(fake_bundle):
    bundle, _ = fake_bundle
    with pytest.raises(ValueError, match="segmentation"):
        read_xenium(bundle, segmentation="bogus", verbose=False)


def test_read_xenium_missing_cells_csv(tmp_path, monkeypatch):
    bundle = tmp_path / "slide_x"
    bundle.mkdir()

    def _fake_read_10x_h5(path):
        a = AnnData(X=np.zeros((1, 1), dtype=np.float32))
        a.obs_names = ["bc"]
        a.var_names = ["g"]
        return a

    monkeypatch.setattr("sccellfie.io.xenium.sc.read_10x_h5", _fake_read_10x_h5)
    (bundle / "cell_feature_matrix.h5").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="cells.csv.gz"):
        read_xenium(bundle, segmentation="cell", verbose=False)


def test_read_xenium_nucleus(tmp_path, monkeypatch):
    bundle = tmp_path / "slide_n"
    bundle.mkdir()

    a = AnnData(X=np.zeros((2, 1), dtype=np.float32))
    a.obs_names = ["bc1", "bc2"]
    a.var_names = ["g"]
    a.obs["x_centroid"] = [1.0, 2.0]
    a.obs["y_centroid"] = [3.0, 4.0]

    def _fake_read_h5ad(path):
        return a

    monkeypatch.setattr("sccellfie.io.xenium.sc.read_h5ad", _fake_read_h5ad)
    (bundle / "nucleus_feature_matrix.h5ad").write_bytes(b"")

    adata = read_xenium(bundle, segmentation="nucleus", verbose=False)
    assert "X_spatial" in adata.obsm
    np.testing.assert_allclose(adata.obsm["X_spatial"], [[1.0, 3.0], [2.0, 4.0]])


def test_read_xenium_slide_id_subdir(tmp_path, monkeypatch):
    root = tmp_path
    bundle = root / "slideX"
    bundle.mkdir()

    def _fake_read_10x_h5(path):
        a = AnnData(X=np.zeros((1, 1), dtype=np.float32))
        a.obs_names = ["bc"]
        a.var_names = ["g"]
        return a

    monkeypatch.setattr("sccellfie.io.xenium.sc.read_10x_h5", _fake_read_10x_h5)
    (bundle / "cell_feature_matrix.h5").write_bytes(b"")
    pd.DataFrame({"x_centroid": [1.0], "y_centroid": [2.0]}).to_csv(
        bundle / "cells.csv.gz", index=False, compression="gzip"
    )

    adata = read_xenium(root, slide_id="slideX", segmentation="cell", verbose=False)
    assert adata.n_obs == 1
