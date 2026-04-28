"""
Reader for 10x Genomics Xenium output bundles.

Builds an :class:`anndata.AnnData` from the cell-feature matrix (or the
nucleus matrix), attaches centroid coordinates under ``obsm[spatial_key]``
(default ``'X_spatial'``, the scCellFie convention), and optionally
attaches the graph-cluster assignments. Cell-boundary polygons are
**not** auto-loaded — call :func:`sccellfie.io.load_xenium_segmentation`
when polygons are needed.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


def _resolve_bundle(data_dir: Union[str, Path], slide_id: Optional[str]) -> Path:
    base = Path(data_dir)
    return base / slide_id if slide_id else base


def _attach_clusters(
    adata: AnnData,
    cluster_path: Path,
    verbose: bool = True,
) -> None:
    clusters = pd.read_csv(cluster_path).set_index("Barcode")
    obs_names = [c for c in adata.obs_names if c in clusters.index]
    if not obs_names:
        if verbose:
            print(f"Note: no overlap between obs_names and {cluster_path.name}")
        return
    clusters = clusters.loc[obs_names, :]
    adata.obs["cluster"] = pd.Categorical(clusters["Cluster"].astype(str))

    def _safe_sort_key(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return float("inf")

    cats = [c for c in adata.obs["cluster"].cat.categories if pd.notna(c)]
    sorted_cats = sorted(cats, key=_safe_sort_key)
    adata.obs["cluster"] = adata.obs["cluster"].cat.reorder_categories(
        sorted_cats, ordered=True
    )


def read_xenium(
    data_dir: Union[str, Path],
    slide_id: Optional[str] = None,
    segmentation: str = "cell",
    cluster_file: Optional[Union[str, Path, bool]] = None,
    spatial_key: str = "X_spatial",
    verbose: bool = True,
) -> AnnData:
    """
    Read a 10x Xenium output bundle into an AnnData.

    Parameters
    ----------
    data_dir : str or Path
        Path to the Xenium bundle root, or to a directory containing one
        sub-directory per slide (in which case ``slide_id`` selects the
        slide).

    slide_id : str, optional (default: None)
        Sub-directory under ``data_dir``. When None, ``data_dir`` itself
        is treated as the bundle root.

    segmentation : {"cell", "nucleus"}, optional (default: "cell")
        ``"cell"`` reads ``cell_feature_matrix.h5`` and joins centroids
        from ``cells.csv.gz``. ``"nucleus"`` reads
        ``nucleus_feature_matrix.h5ad`` and pulls centroids from its
        ``obs`` columns ``x_centroid`` / ``y_centroid``.

    cluster_file : str, Path, False, or None, optional (default: None)
        Path to a cluster-assignment CSV (with columns ``Barcode``,
        ``Cluster``). When None, ``analysis/clustering/gene_expression_graphclust/clusters.csv``
        is auto-loaded if present. Pass ``False`` to skip the lookup.

    spatial_key : str, optional (default: "X_spatial")
        Key under which centroids are stored in ``adata.obsm``. Defaults
        to scCellFie's canonical key; pass ``"spatial"`` if you also want
        ``scanpy.pl.spatial`` to find them.

    verbose : bool, optional (default: True)
        Print informational messages.

    Returns
    -------
    anndata.AnnData
        AnnData with centroid coordinates in ``adata.obsm[spatial_key]``
        and any cluster assignments in ``adata.obs['cluster']``.
    """
    bundle = _resolve_bundle(data_dir, slide_id)

    if segmentation == "cell":
        adata = sc.read_10x_h5(bundle / "cell_feature_matrix.h5")
        metadata_path = bundle / "cells.csv.gz"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Expected cells.csv.gz at {metadata_path}; required for "
                "centroid coordinates when segmentation='cell'."
            )
        metadata = pd.read_csv(metadata_path)
        metadata.set_index(adata.obs_names, inplace=True)
        adata.obsm[spatial_key] = (
            metadata[["x_centroid", "y_centroid"]].copy().to_numpy()
        )
        adata.obs = metadata.drop(columns=["x_centroid", "y_centroid"])
    elif segmentation == "nucleus":
        adata = sc.read_h5ad(bundle / "nucleus_feature_matrix.h5ad")
        if not {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
            raise ValueError(
                "nucleus_feature_matrix.h5ad must have 'x_centroid' and "
                "'y_centroid' columns in obs."
            )
        adata.obsm[spatial_key] = adata.obs[["x_centroid", "y_centroid"]].to_numpy()
    else:
        raise ValueError(
            f"segmentation must be 'cell' or 'nucleus', got {segmentation!r}"
        )

    if cluster_file is False:
        return adata

    if cluster_file is None:
        cluster_path = bundle / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv"
        if cluster_path.exists():
            _attach_clusters(adata, cluster_path, verbose=verbose)
        elif verbose:
            print(f"Note: {cluster_path} not found, skipping cluster attachment.")
    else:
        cluster_path = Path(cluster_file)
        if not cluster_path.exists():
            raise FileNotFoundError(f"cluster_file not found: {cluster_path}")
        _attach_clusters(adata, cluster_path, verbose=verbose)

    return adata
