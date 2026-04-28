"""
Cell segmentation polygon loaders.

Generic loader for vertex-table segmentation files (Xenium parquet,
CSV/TSV, optionally gzipped) plus convenience wrappers. Column names
are auto-detected when not supplied. ``geopandas`` and ``shapely`` are
imported lazily so users who only score non-spatial data don't need
them installed.
"""
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def _require_geo():
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon, MultiPolygon
    except ImportError as e:
        raise ImportError(
            "Loading cell segmentations requires geopandas and shapely. "
            "Install via: pip install geopandas shapely"
        ) from e
    return gpd, Polygon, MultiPolygon


def load_segmentation(
    filepath: str,
    cell_ids: Optional[np.ndarray] = None,
    cell_id_col: Optional[str] = None,
    vertex_x_col: Optional[str] = None,
    vertex_y_col: Optional[str] = None,
    output: str = "geodataframe",
) -> Union["gpd.GeoDataFrame", dict]:
    """
    Load cell boundary polygons from a segmentation file.

    Generic loader for any vertex-table format (one row per polygon
    vertex). Supports Xenium parquet, CSV.gz, CSV, TSV, and TSV.gz with
    auto-detection of column names.

    Parameters
    ----------
    filepath : str
        Path to the cell boundaries file. Accepted extensions are
        ``.parquet``, ``.csv.gz``, ``.csv``, ``.tsv``, and ``.tsv.gz``.

    cell_ids : np.ndarray, optional (default: None)
        If provided, only load polygons for these cell IDs.

    cell_id_col : str, optional (default: None)
        Column name for cell identifiers. Auto-detected if None. Tries
        ``"cell_id"``, ``"ID"``, ``"id"``, ``"cell_ID"`` in that order.

    vertex_x_col : str, optional (default: None)
        Column name for vertex x-coordinates. Auto-detected if None.
        Tries ``"vertex_x"``, ``"x_location"``, ``"X"``.

    vertex_y_col : str, optional (default: None)
        Column name for vertex y-coordinates. Auto-detected if None.
        Tries ``"vertex_y"``, ``"y_location"``, ``"Y"``.

    output : {"geodataframe", "dict"}, optional (default: "geodataframe")
        Return format. ``"geodataframe"`` returns a GeoDataFrame indexed
        by cell ID with ``centroid_x`` / ``centroid_y`` columns.
        ``"dict"`` returns a mapping of ``cell_id -> shapely.Polygon``.

    Returns
    -------
    geopandas.GeoDataFrame or dict
        Cell boundary polygons in the requested format.
    """
    gpd, Polygon, MultiPolygon = _require_geo()

    lower = filepath.lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    elif lower.endswith(".csv.gz") or lower.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif lower.endswith(".tsv.gz") or lower.endswith(".tsv"):
        df = pd.read_csv(filepath, sep="\t")
    else:
        raise ValueError(
            f"Unsupported file format: {filepath}. "
            "Expected .parquet, .csv, .csv.gz, .tsv, or .tsv.gz"
        )

    if cell_id_col is None:
        for candidate in ("cell_id", "ID", "id", "cell_ID"):
            if candidate in df.columns:
                cell_id_col = candidate
                break
        if cell_id_col is None:
            raise ValueError(
                f"Could not find cell ID column. Available: {df.columns.tolist()}"
            )

    if vertex_x_col is None:
        for candidate in ("vertex_x", "x_location", "X"):
            if candidate in df.columns:
                vertex_x_col = candidate
                break
        if vertex_x_col is None:
            raise ValueError(
                f"Could not find x-coordinate column. Available: {df.columns.tolist()}"
            )

    if vertex_y_col is None:
        for candidate in ("vertex_y", "y_location", "Y"):
            if candidate in df.columns:
                vertex_y_col = candidate
                break
        if vertex_y_col is None:
            raise ValueError(
                f"Could not find y-coordinate column. Available: {df.columns.tolist()}"
            )

    if cell_ids is not None:
        df = df[df[cell_id_col].isin(cell_ids)]

    polygons = {}
    for cid, group in tqdm(
        df.groupby(cell_id_col), desc="Loading segmentation", leave=False
    ):
        coords = group[[vertex_x_col, vertex_y_col]].values
        if len(coords) < 3:
            continue
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            polygons[cid] = poly
        except Exception:
            continue

    if output == "dict":
        return polygons

    gdf = gpd.GeoDataFrame(
        geometry=list(polygons.values()),
        index=pd.Index(polygons.keys(), name=cell_id_col),
    )
    gdf["centroid_x"] = gdf.geometry.centroid.x
    gdf["centroid_y"] = gdf.geometry.centroid.y
    return gdf


def load_xenium_segmentation(
    filepath: str,
    cell_ids: Optional[np.ndarray] = None,
    cell_id_col: Optional[str] = None,
    vertex_x_col: Optional[str] = None,
    vertex_y_col: Optional[str] = None,
    output: str = "geodataframe",
) -> Union["gpd.GeoDataFrame", dict]:
    """
    Load cell boundaries from a Xenium ``cell_boundaries`` file.

    Thin wrapper around :func:`load_segmentation` kept for discoverability.
    Xenium ``cell_boundaries`` files use the default auto-detected columns
    (``cell_id``, ``vertex_x``, ``vertex_y``) so this is equivalent to
    calling :func:`load_segmentation` directly.

    See :func:`load_segmentation` for parameter and return documentation.
    """
    return load_segmentation(
        filepath=filepath,
        cell_ids=cell_ids,
        cell_id_col=cell_id_col,
        vertex_x_col=vertex_x_col,
        vertex_y_col=vertex_y_col,
        output=output,
    )


def load_segmentation_from_gdf(gdf, geometry_col: str = "geometry"):
    """
    Prepare a pre-loaded GeoDataFrame for downstream plotting.

    Adds ``centroid_x`` and ``centroid_y`` columns if missing.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with polygon geometries.

    geometry_col : str, optional (default: "geometry")
        Name of the geometry column.

    Returns
    -------
    geopandas.GeoDataFrame
        Input GeoDataFrame with centroid columns added.
    """
    _require_geo()

    if geometry_col != "geometry":
        gdf = gdf.set_geometry(geometry_col)

    if "centroid_x" not in gdf.columns:
        gdf["centroid_x"] = gdf.geometry.centroid.x
    if "centroid_y" not in gdf.columns:
        gdf["centroid_y"] = gdf.geometry.centroid.y

    return gdf
