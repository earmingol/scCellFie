import gzip

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon

from sccellfie.io import (
    load_segmentation,
    load_xenium_segmentation,
    load_segmentation_from_gdf,
)


def _square_vertices(cell_id, x0, y0, side=1.0):
    return pd.DataFrame(
        {
            "cell_id": [cell_id] * 4,
            "vertex_x": [x0, x0 + side, x0 + side, x0],
            "vertex_y": [y0, y0, y0 + side, y0 + side],
        }
    )


def _make_vertex_table():
    return pd.concat(
        [
            _square_vertices("A", 0.0, 0.0),
            _square_vertices("B", 5.0, 0.0),
            _square_vertices("C", 0.0, 5.0, side=2.0),
        ],
        ignore_index=True,
    )


def test_load_segmentation_csv_geodataframe(tmp_path):
    df = _make_vertex_table()
    path = tmp_path / "boundaries.csv"
    df.to_csv(path, index=False)

    out = load_segmentation(str(path))

    assert isinstance(out, gpd.GeoDataFrame)
    assert sorted(out.index.tolist()) == ["A", "B", "C"]
    np.testing.assert_allclose(out.loc["A", "centroid_x"], 0.5)
    np.testing.assert_allclose(out.loc["A", "centroid_y"], 0.5)
    np.testing.assert_allclose(out.loc["C", "centroid_x"], 1.0)
    np.testing.assert_allclose(out.loc["C", "centroid_y"], 6.0)


def test_load_segmentation_dict_output(tmp_path):
    df = _make_vertex_table()
    path = tmp_path / "boundaries.csv"
    df.to_csv(path, index=False)

    out = load_segmentation(str(path), output="dict")

    assert isinstance(out, dict)
    assert set(out.keys()) == {"A", "B", "C"}
    for poly in out.values():
        assert poly.is_valid


def test_load_segmentation_filter_cell_ids(tmp_path):
    df = _make_vertex_table()
    path = tmp_path / "boundaries.csv"
    df.to_csv(path, index=False)

    out = load_segmentation(str(path), cell_ids=np.array(["A", "C"]))
    assert sorted(out.index.tolist()) == ["A", "C"]


def test_load_segmentation_gz_csv(tmp_path):
    df = _make_vertex_table()
    path = tmp_path / "boundaries.csv.gz"
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, index=False)

    out = load_segmentation(str(path), output="dict")
    assert set(out.keys()) == {"A", "B", "C"}


def test_load_segmentation_tsv(tmp_path):
    df = _make_vertex_table()
    path = tmp_path / "boundaries.tsv"
    df.to_csv(path, index=False, sep="\t")

    out = load_segmentation(str(path), output="dict")
    assert set(out.keys()) == {"A", "B", "C"}


def test_load_segmentation_atera_columns(tmp_path):
    df = _make_vertex_table().rename(
        columns={"cell_id": "ID", "vertex_x": "x_location", "vertex_y": "y_location"}
    )
    path = tmp_path / "boundaries.csv"
    df.to_csv(path, index=False)

    out = load_segmentation(str(path), output="dict")
    assert set(out.keys()) == {"A", "B", "C"}


def test_load_segmentation_explicit_columns(tmp_path):
    df = _make_vertex_table().rename(
        columns={"cell_id": "my_cell", "vertex_x": "vx", "vertex_y": "vy"}
    )
    path = tmp_path / "boundaries.csv"
    df.to_csv(path, index=False)

    out = load_segmentation(
        str(path),
        cell_id_col="my_cell",
        vertex_x_col="vx",
        vertex_y_col="vy",
        output="dict",
    )
    assert set(out.keys()) == {"A", "B", "C"}


def test_load_segmentation_unknown_columns_errors(tmp_path):
    df = pd.DataFrame({"a": [0, 0], "b": [1, 1], "c": [2, 2]})
    path = tmp_path / "boundaries.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="cell ID"):
        load_segmentation(str(path))


def test_load_segmentation_unsupported_extension(tmp_path):
    path = tmp_path / "boundaries.xyz"
    path.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_segmentation(str(path))


def test_load_xenium_segmentation_alias(tmp_path):
    df = _make_vertex_table()
    path = tmp_path / "boundaries.csv"
    df.to_csv(path, index=False)

    a = load_xenium_segmentation(str(path), output="dict")
    b = load_segmentation(str(path), output="dict")
    assert set(a.keys()) == set(b.keys())


def test_load_segmentation_from_gdf_adds_centroids():
    polys = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    gdf = gpd.GeoDataFrame(geometry=polys)
    out = load_segmentation_from_gdf(gdf)
    assert "centroid_x" in out.columns
    assert "centroid_y" in out.columns
    np.testing.assert_allclose(out["centroid_x"].iloc[0], 0.5)
    np.testing.assert_allclose(out["centroid_y"].iloc[0], 0.5)
