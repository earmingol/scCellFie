import geopandas as gpd
import numpy as np
import pandas as pd

from esda.getisord import G_Local
from libpysal.weights.contiguity import Queen
from shapely.geometry import Point

from scipy.sparse import issparse
from tqdm import tqdm


def obtain_hotspots(adata, spatial_key='X_spatial', use_raw=False):
    """
    Obtains hotspots for each variable in adata.var_names.
    It uses the Getis-Ord Gi* statistic from PySAL.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    spatial_key: str, optional (default: 'X_spatial')
        The key in adata.obsm where the spatial coordinates are stored.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    hotspots: dict
        A dictionary with variable names as keys and Getis-Ord Gi* statistics as values.
    """
    assert spatial_key in adata.obsm.keys(), "obsm_key must be a key in adata.obsm."

    hotspots = dict()

    geometries = [Point(p[0], p[1]) for p in adata.obsm[spatial_key]]

    for var_ in tqdm(adata.var_names):
        if use_raw:
            X = adata.raw.X
        else:
            X = adata.X

        if issparse(X):
            X = X.toarray()

        var_idx = np.argwhere(adata.var_names == var_).item()

        gdf = gpd.GeoDataFrame({'intensity': X[:, var_idx].flatten()},
                               geometry=geometries)  # geometries represent pixel locations
        w = Queen.from_dataframe(gdf, use_index=False)
        w.transform = 'r'

        gi_star = G_Local(gdf['intensity'], w)
        hotspots[var_] = gi_star
    return hotspots


def calculate_aggregate_metric(z_scores, z_threshold=1.96):
    """
    Aggregates the Getis-Ord Gi* statistics for a variable.

    Parameters
    ----------
    z_scores: numpy.ndarray
        The z-scores of the variable.

    z_threshold: float, optional (default: 1.96)
        The z-score threshold to use for determining hotspots.

    Returns
    -------
    mean_z: float
        The mean z-score of the variable.

    median_z: float
        The median z-score of the variable.

    hotspot_proportion: float
        The proportion of hotspots for the variable.

    coldspot_proportion: float
        The proportion of coldspots for the variable.

    significant_proportion: float
        The proportion of significant pixels for the variable.
    """
    mean_z = np.mean(z_scores)
    median_z = np.median(z_scores)
    hotspot_proportion = np.sum(z_scores > z_threshold) / len(z_scores)
    coldspot_proportion = np.sum(-1 * z_scores > z_threshold) / len(z_scores)
    significant_proportion = np.sum(np.abs(z_scores) > z_threshold) / len(z_scores)
    return mean_z, median_z, hotspot_proportion, coldspot_proportion, significant_proportion


def summarize_hotspots(hotspots, z_threshold=1.96):
    """
    Summarizes hotspots for each variable in adata.var_names.
    It computes the mean, median, and proportion of hotspots, coldspots, and significant pixels.

    Parameters
    ----------
    hotspots: dict
        A dictionary with variable names as keys and Getis-Ord Gi* statistics as values.

    z_threshold: float, optional (default: 1.96)
        The z-score threshold to use for determining hotspots.

    Returns
    -------
    hotspot_df: pandas.DataFrame
        A pandas.DataFrame object containing the aggregate metrics for each variable in adata.var_names.

    Notes
    -----
    Columns in hotspot_df:
        Var-Name: The variable name.
        Mean-Hotspot-Z: The mean z-score of the variable.
        Median-Hotspot-Z: The median z-score of the variable.
        Hotspot-Proportion: The proportion of hotspots for the variable.
        Coldspot-Proportion: The proportion of coldspots for the variable.
        Significant-Proportion: The proportion of significant pixels for the variable.
    """
    records = []

    for k, v in hotspots.items():
        z_scores = v.Zs
        mean_z, median_z, hotspot_proportion, coldspot_proportion, significant_proportion = calculate_aggregate_metric(z_scores, z_threshold=z_threshold)
        records.append((k, mean_z, median_z, hotspot_proportion, coldspot_proportion, significant_proportion))

    hotspot_df = pd.DataFrame.from_records(records, columns=['Var-Name', 'Mean-Hotspot-Z', 'Median-Hotspot-Z',
                                                             'Hotspot-Proportion', 'Coldspot-Proportion',
                                                             'Significant-Proportion'])
    return hotspot_df


def compute_hotspots(adata, spatial_key='X_spatial', use_raw=False, z_threshold=1.96):
    """
    Computes hotspots for each variable in adata.var_names.
    Hotspots are computed using the Getis-Ord Gi* statistic.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    spatial_key: str, optional (default: 'X_spatial')
        The key in adata.obsm where the spatial coordinates are stored.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    z_threshold: float, optional (default: 1.96)
        The z-score threshold to use for determining hotspots.

    Returns
    -------
    None
        - A dictionary containing the Getis-Ord Gi* statistic for each variable in adata.var_names
        is added to adata.uns['hotspots']['hotspots'].

        - A pandas.DataFrame object containing the aggregate metrics for each variable in adata.var_names
        is added to adata.uns['hotspots']['hotspot_df'].
    """
    hotspots = obtain_hotspots(adata, spatial_key=spatial_key, use_raw=use_raw)
    hotspot_df = summarize_hotspots(hotspots, z_threshold=z_threshold)
    adata.uns['hotspots'] = {'hotspots': hotspots, 'hotspot_df': hotspot_df}