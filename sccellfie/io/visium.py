"""
Reader for 10x Visium and VisiumHD (segmented) bundles.

Standard Visium and VisiumHD-bins layouts are delegated to
:func:`scanpy.read_visium`. The VisiumHD-segmented layout
(``cell_segmentations.geojson`` + ``filtered_feature_bc_matrix.h5``)
has a custom branch that derives centroids and per-cell areas from
the cell polygons, optionally folds in nucleus areas, and stores
coordinates under both ``obsm['spatial']`` (scanpy convention) and
``obsm['X_spatial']`` (scCellFie convention).
"""
import json
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from h5py import File
from matplotlib.image import imread
from scanpy import read_10x_h5


def _detect_hd_layout(path: Path) -> str:
    if (path / "cell_segmentations.geojson").exists():
        return "segmented"
    if (path / "spatial" / "tissue_positions.parquet").exists():
        return "bins"
    return "standard"


def _read_visium_segmented(
    path: Path,
    count_file: str,
    library_id: Optional[str],
    source_image_path: Optional[Union[str, Path]],
    genome: Optional[str],
    load_images: bool,
) -> AnnData:
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "Reading VisiumHD-segmented bundles requires geopandas. "
            "Install via: pip install geopandas shapely"
        ) from e

    adata = read_10x_h5(path / count_file, genome=genome)
    adata.uns["spatial"] = {}

    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)
    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")

    adata.uns["spatial"][library_id] = {}

    files = {
        "scalefactors_json_file": path / "spatial" / "scalefactors_json.json",
        "hires_image": path / "spatial" / "tissue_hires_image.png",
        "lowres_image": path / "spatial" / "tissue_lowres_image.png",
    }
    if not files["scalefactors_json_file"].exists():
        raise OSError(f"Could not find {files['scalefactors_json_file']}")

    if load_images:
        adata.uns["spatial"][library_id]["images"] = {}
        for res in ("hires", "lowres"):
            img_path = files[f"{res}_image"]
            if img_path.exists():
                adata.uns["spatial"][library_id]["images"][res] = imread(str(img_path))
            else:
                warnings.warn(f"Could not find {img_path}", stacklevel=2)

    adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
        files["scalefactors_json_file"].read_bytes()
    )
    adata.uns["spatial"][library_id]["metadata"] = {
        k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
        for k in ("chemistry_description", "software_version")
        if k in attrs
    }

    geojson_path = path / "cell_segmentations.geojson"
    if not geojson_path.exists():
        raise OSError(f"Could not find cell_segmentations.geojson at {geojson_path}")
    cell_boundaries = gpd.read_file(geojson_path)
    cell_boundaries["cell_id"] = [
        f"cellid_{int(cid):09d}-1" for cid in cell_boundaries["cell_id"].values
    ]

    cell_areas_pixels = cell_boundaries.geometry.area
    avg_area_pixels = float(np.mean(cell_areas_pixels))

    microns_to_pixel = adata.uns["spatial"][library_id]["scalefactors"].get(
        "microns_per_pixel", 1.0
    )
    bin_size_um = (2.0 * np.sqrt(avg_area_pixels) / np.sqrt(np.pi)) * microns_to_pixel
    spot_diameter_fullres = bin_size_um * 4.521168684149367
    adata.uns["spatial"][library_id]["scalefactors"]["bin_size_um"] = bin_size_um
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres

    centroids = cell_boundaries.geometry.centroid
    positions = pd.DataFrame(
        {
            "barcode": cell_boundaries["cell_id"],
            "cell_area": cell_areas_pixels * (microns_to_pixel ** 2),
            "in_tissue": 1,
            "pxl_row_in_fullres": centroids.x,
            "pxl_col_in_fullres": centroids.y,
        }
    )

    nucleus_geojson_path = path / "nucleus_segmentations.geojson"
    if nucleus_geojson_path.exists():
        try:
            nucleus_boundaries = gpd.read_file(nucleus_geojson_path)
            nucleus_boundaries["cell_id"] = [
                f"cellid_{int(cid):09d}-1"
                for cid in nucleus_boundaries["cell_id"].values
            ]
            nucleus_df = pd.DataFrame(
                {
                    "barcode": nucleus_boundaries["cell_id"],
                    "nucleus_area": nucleus_boundaries.geometry.area
                    * (microns_to_pixel ** 2),
                }
            )
            positions = positions.merge(nucleus_df, on="barcode", how="left")
        except Exception as e:
            warnings.warn(f"Could not load nucleus segmentation data: {e}", stacklevel=2)

    positions.set_index("barcode", inplace=True)
    adata.obs = adata.obs.join(positions, how="left")
    adata.obsm["spatial"] = adata.obs[
        ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    ].to_numpy()
    adata.obsm["X_spatial"] = adata.obsm["spatial"].copy()
    adata.obs.drop(
        columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True
    )

    if source_image_path is not None:
        adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
            Path(source_image_path).resolve()
        )

    adata.uns["spatial"][library_id]["metadata"]["visium_hd"] = True
    adata.uns["spatial"][library_id]["metadata"]["hd_layout"] = "segmented"
    return adata


def read_visium(
    path: Union[str, Path],
    *,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: Optional[str] = None,
    source_image_path: Optional[Union[str, Path]] = None,
    is_hd: bool = False,
    hd_layout: str = "detect",
    genome: Optional[str] = None,
    load_images: bool = True,
) -> AnnData:
    """
    Read a 10x Visium / VisiumHD bundle into an AnnData.

    Standard Visium and VisiumHD-bins layouts delegate to
    :func:`scanpy.read_visium`. The VisiumHD-segmented layout (presence
    of ``cell_segmentations.geojson``) is handled by a custom branch
    that derives centroids and per-cell areas from the polygons,
    optionally merges nucleus areas from
    ``nucleus_segmentations.geojson``, and writes coordinates under
    both ``obsm['spatial']`` and ``obsm['X_spatial']``.

    Parameters
    ----------
    path : str or Path
        Path to the Visium bundle directory.

    count_file : str, optional (default: "filtered_feature_bc_matrix.h5")
        Filename of the count matrix inside ``path``.

    library_id : str, optional (default: None)
        Identifier used as the key under ``adata.uns['spatial']``. When
        None it is read from the count file's HDF5 attributes.

    source_image_path : str or Path, optional (default: None)
        Path to the high-resolution tissue image, recorded under
        ``adata.uns['spatial'][library_id]['metadata']['source_image_path']``.

    is_hd : bool, optional (default: False)
        Whether this is a VisiumHD bundle. Used together with
        ``hd_layout`` to dispatch to the right branch.

    hd_layout : {"detect", "bins", "segmented", "standard"}, optional (default: "detect")
        Force a specific HD layout. ``"detect"`` (default) auto-detects:
        ``"segmented"`` if ``cell_segmentations.geojson`` is present,
        otherwise ``"bins"`` if ``spatial/tissue_positions.parquet`` is
        present, otherwise ``"standard"``.

    genome : str, optional (default: None)
        Filter expression to genes within this genome (passed through
        to :func:`scanpy.read_visium`).

    load_images : bool, optional (default: True)
        Whether to load hires/lowres tissue images.

    Returns
    -------
    anndata.AnnData
        AnnData with spatial information stored in standard scanpy
        format. For the segmented branch, also exposes
        ``adata.obsm['X_spatial']`` (scCellFie convention).
    """
    path = Path(path)

    layout = hd_layout
    if is_hd and layout == "detect":
        layout = _detect_hd_layout(path)
    elif not is_hd:
        layout = "standard"

    if layout == "segmented":
        return _read_visium_segmented(
            path=path,
            count_file=count_file,
            library_id=library_id,
            source_image_path=source_image_path,
            genome=genome,
            load_images=load_images,
        )

    adata = sc.read_visium(
        path,
        genome=genome,
        count_file=count_file,
        library_id=library_id,
        load_images=load_images,
        source_image_path=source_image_path,
    )
    if "spatial" in adata.obsm and "X_spatial" not in adata.obsm:
        adata.obsm["X_spatial"] = np.asarray(adata.obsm["spatial"]).copy()
    return adata
