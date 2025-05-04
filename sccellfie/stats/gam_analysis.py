import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from sccellfie.expression.aggregation import agg_expression_cells
from sccellfie.preprocessing.matrix_utils import get_matrix_gene_expression


def generate_pseudobulks(adata, cell_type_key, n_pseudobulks=5, cells_per_bulk=1000,
                         layer=None, use_raw=False, genes=None, agg_func='trimean',
                         continuous_key=None, random_seed=None):
    """
    Generates pseudo-bulk samples from single-cell data. Each pseudo-bulk
    represents a group of cells from the same cell type.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing the single-cell expression data.

    cell_type_key : str
        The key in adata.obs that contains the cell type annotations.

    n_pseudobulks : int, optional (default: 5)
        The number of pseudo-bulk samples to generate for each cell type.
        Less will be generated if there are fewer cells than
        the n_pseudobulks * cells_per_bulk.

    cells_per_bulk : int, optional (default: 1000)
        The number of cells to include in each pseudo-bulk sample. Less
        will be used if there are fewer cells in the cell type.

    layer : str, optional (default: None)
        The name of the layer in adata to use for aggregation. If None,
        the main expression matrix adata.X is used.

    use_raw : bool, optional (default: False)
        Whether to use the data in adata.raw.X (True) or in adata.X (False).

    genes : list, optional (default: None)
        List of gene names to include in the pseudo-bulk samples. If None,
        all genes are included.

    agg_func : str, optional (default: 'trimean')
        The aggregation function to apply. Options are 'mean', 'median',
        '25p' (25th percentile), '75p' (75th percentile), 'trimean' (0.5*Q2 + 0.25(Q1+Q3)),
        and 'topmean' (computed among the top `top_percent`% of values).

    continuous_key : str, optional (default: None)
        The key in adata.obs that contains continuous values to include in the
        pseudo-bulk samples. If None, continuous values are not included. This
        is useful for trajectory analysis or other continuous annotations.

    random_seed : int, optional (default: None)
        Random seed for reproducible pseudo-bulk generation.

    Returns
    -------
    adata_pseudobulk : AnnData
        An AnnData object containing the pseudo-bulk samples. The expression
        values are aggregated across the cells in each pseudo-bulk. The obs
        DataFrame contains the cell type annotations and the continuous values
        if provided.
    """
    # Generate pseudo-bulk assignments
    pseudobulk_ids = pd.Series(index=adata.obs.index, dtype=str)

    # Store continuous values for each pseudo-bulk if needed
    continuous_values = {}
    valid_pseudobulks = []  # Keep track of valid pseudobulks

    # Get available cells for each cell type
    available_cells = {ct: set(adata.obs.index[adata.obs[cell_type_key] == ct])
                       for ct in adata.obs[cell_type_key].unique()}

    # Create RNG instance for reproducibility
    rng = np.random.RandomState(random_seed)

    for cell_type in adata.obs[cell_type_key].unique():
        ct_cells = available_cells[cell_type]

        if len(ct_cells) < cells_per_bulk:
            print(f"Warning: Cell type {cell_type} has fewer than {cells_per_bulk} cells.")
            possible_pseudobulks = 1 # Only generate one pseudobulk for this cell type
        else:
            possible_pseudobulks = min(math.ceil(len(ct_cells) / cells_per_bulk), n_pseudobulks) # Respect the maximum allowed pseudobulks

        for i in range(possible_pseudobulks):
            bulk_id = f"{cell_type}_bulk_{i + 1}"
            valid_pseudobulks.append(bulk_id)

            # Select cells for this pseudo-bulk
            selected_cells = rng.choice(  # Using rng instead of np.random
                list(ct_cells),
                size=min(cells_per_bulk, len(ct_cells)),  # Ensure we don't try to select more cells than available
                replace=False #len(ct_indices) < cells_per_bulk
            )

            pseudobulk_ids.loc[selected_cells] = bulk_id
            ct_cells.difference_update(selected_cells) # Remove selected cells from available cells for next pseudobulks

            # If continuous key is provided, calculate mean for these cells
            if continuous_key is not None:
                continuous_values[bulk_id] = adata.obs.loc[selected_cells, continuous_key].mean()

    # Create pseudo-bulk data
    adata.obs['pseudobulk_id'] = pseudobulk_ids
    pseudobulk_data = agg_expression_cells(
        adata[~pseudobulk_ids.isna()],  # Only use cells that were assigned to pseudobulks
        groupby='pseudobulk_id',
        layer=layer,
        gene_symbols=genes,
        agg_func=agg_func,
        use_raw=use_raw
    )

    # Extract cell types from pseudo-bulk IDs
    bulk_cell_types = pd.Series(index=pseudobulk_data.index, dtype=str)
    for idx in pseudobulk_data.index:
        bulk_cell_types[idx] = idx.split('_bulk_')[0]

    # Create the obs DataFrame for the pseudo-bulk AnnData
    obs_df = pd.DataFrame({cell_type_key: bulk_cell_types}, index=pseudobulk_data.index)

    # Add continuous values if they were calculated
    if continuous_key is not None:
        obs_df[continuous_key] = pd.Series(continuous_values)

    # Create pseudo-bulk AnnData
    from anndata import AnnData
    adata_pseudobulk = AnnData(
        X=pseudobulk_data.values,
        obs=obs_df,
        var=pd.DataFrame(index=pseudobulk_data.columns)
    )

    return adata_pseudobulk, pseudobulk_ids


def fit_gam_model(adata, cell_type_key, cell_type_order=None, continuous_key=None, genes=None,
                  layer=None, use_raw=False, n_splines=10, spline_order=3, lam=0.6, normalize=False, use_pseudobulk=False, n_pseudobulks=5,
                  cells_per_bulk=1000, pseudobulk_agg='trimean', **kwargs):
    """
    Fits Generalized Additive Models (GAMs) to single-cell data for each gene.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing the single-cell expression data.

    cell_type_key : str
        The key in adata.obs that contains the cell type annotations.

    cell_type_order : list, optional (default: None)
        The order in which to process cell types. If None, cell types are
        processed in alphabetical order. This is useful when you have a
        known biological order for the cell types.

    continuous_key : str, optional (default: None)
        The key in adata.obs that contains continuous values to include in the
        GAM models. If None, continuous values are not included. This is useful
        for trajectory analysis or other continuous annotations.

    genes : list, optional (default: None)
        List of gene names to include in the GAM models. If None, all genes are included.

    layer : str, optional (default: None)
        The name of the layer in adata to use for aggregation. If None,
        the main expression matrix adata.X is used.

    use_raw : bool, optional (default: False)
        Whether to use the data in adata.raw.X (True) or in adata.X (False).

    n_splines : int, optional (default: 10)
        Number of splines to use for the feature function in the GAM. Must be non-negative.

    spline_order : int, optional (default: 3)
        Order of spline to use for the feature function in the GAM. Must be non-negative.

    lam : float, optional (default: 0.6)
        Strength of smoothing penalty in the GAM. Must be a positive float.
        Larger values enforce stronger smoothing.

    normalize : bool, optional (default: False)
        Whether to normalize the expression values for each gene. This normalization
        is of the type min-max scaling, where the minimum and maximum values are 0 and 1.

    use_pseudobulk : bool, optional (default: False)
        Whether to use pseudobulk samples for the GAM analysis. If True, the GAM models
        are fitted to the aggregated expression values for each cell type. This is useful
        for reducing the biased on the statistical power due to having many single cells.

    n_pseudobulks : int, optional (default: 5)
        The number of pseudo-bulk samples to generate for each cell type.

    cells_per_bulk : int, optional (default: 1000)
        The number of cells to include in each pseudo-bulk sample.

    pseudobulk_agg : str, optional (default: 'trimean')
        The aggregation function to apply when generating the pseudo-bulk samples.
        Options are 'mean', 'median', '25p' (25th percentile), '75p' (75th percentile),
        'trimean' (0.5*Q2 + 0.25(Q1+Q3)), and 'topmean' (computed among the top `top_percent`% of values).

    kwargs : dict, optional
        Additional keyword arguments to pass to the GAM model. You can find more about it
        in the pygam documentation: https://pygam.readthedocs.io/en/latest/api/gam.html.

    Returns
    -------
    result : dict
        A dictionary containing the fitted GAM models, the model scores, and additional
        information about the pseudo-bulk assignments and cell type encoder when applicable.
    """
    try:
        from pygam import GAM, s
    except ImportError:
        raise ImportError(
            "The pygam package is required for GAM analysis. "
            "Please install it using:\n"
            "pip install pygam\n"
            "or\n"
            "conda install -c conda-forge pygam"
        )

    if use_pseudobulk:
        adata_use, pseudobulk_ids = generate_pseudobulks(
            adata,
            cell_type_key=cell_type_key,
            n_pseudobulks=n_pseudobulks,
            cells_per_bulk=cells_per_bulk,
            layer=layer,
            use_raw=use_raw,
            genes=genes,
            agg_func=pseudobulk_agg,
            continuous_key=continuous_key
        )
    else:
        adata_use = adata

    # Get the expression matrix
    if use_raw and adata_use.raw is not None:
        matrix = adata_use.raw.X
        var_names = adata_use.raw.var_names
    else:
        var_names = adata_use.var_names
        if layer is not None:
            matrix = adata_use.layers[layer]
        else:
            matrix = adata_use.X

    # Filter and order cell types
    preserve_order = False
    if cell_type_order is not None:
        cell_filter = adata_use.obs[cell_type_key].isin(cell_type_order)
        matrix = matrix[cell_filter, :]
        cell_types_series = adata_use.obs[cell_type_key][cell_filter]
        cell_type_order = pd.Categorical(cell_types_series, categories=cell_type_order, ordered=True)
        preserve_order = True
    else:
        cell_types_series = adata_use.obs[cell_type_key]
        cell_type_order = cell_types_series

    # Initialize le here so it's always defined
    le = None
    if continuous_key is not None:
        X = adata_use.obs[continuous_key].rank(method='min').astype(int).values.reshape(-1, 1)
    elif preserve_order:
        X = cell_type_order.codes.reshape(-1, 1)
    else:
        # Encode cell types alphabetically
        le = LabelEncoder()
        X = le.fit_transform(cell_type_order).reshape(-1, 1)

    # Prepare gene list
    if genes is None:
        genes = var_names.tolist()
    else:
        genes = [g for g in genes if g in var_names]

    # Fit models
    models = {}
    scores = {}

    for gene in tqdm(genes, desc='Fitting GAMs for each var in adata'):
        try:
            y = get_matrix_gene_expression(matrix, var_names, gene, normalize=normalize)
            gam = GAM(s(0, n_splines=n_splines, basis='ps', spline_order=spline_order, lam=lam), **kwargs)
            gam.fit(X, y)

            models[gene] = gam
            scores[gene] = {
                'n_samples': gam.statistics_['n_samples'],
                'edof': gam.statistics_['edof'],
                'scale': gam.statistics_['scale'],
                'AIC': gam.statistics_['AIC'],
                'loglikelihood': gam.statistics_['loglikelihood'],
                'deviance': gam.statistics_['deviance'],
                'p_value': gam.statistics_['p_values'][0],
                'explained_deviance': gam.statistics_['pseudo_r2']['explained_deviance'],
                'mcfadden_r2': gam.statistics_['pseudo_r2']['McFadden'],
                'mcfadden_r2_adj': gam.statistics_['pseudo_r2']['McFadden_adj']
            }

        except Exception as e:
            print(f"Failed to fit GAM for gene {gene}:")
            print(f"Error: {str(e)}")
            continue

    if not models:
        raise ValueError("No successful GAM fits. Check your data and parameters.")

    result = {
        'models': models,
        'scores': pd.DataFrame(scores).T,
        'pseudobulk_assignments': None,
        'cell_type_encoder': None
    }

    if use_pseudobulk:
        result['pseudobulk_assignments'] = pseudobulk_ids
    elif le is not None:
        result['cell_type_encoder'] = le

    return result


def analyze_gam_results(gam_results, significance_threshold=0.05, fdr_level=0.05):
    """
    Analyzes GAM model results with FDR correction using statsmodels.

    Parameters
    ----------
    gam_results : dict
        A dictionary containing the results of the GAM analysis. It should
        contain the 'scores' key with a DataFrame of model scores for each gene.

    significance_threshold : float, optional (default: 0.05)
        The significance threshold to consider a gene as significant.

    fdr_level : float, optional (default: 0.05)
        The False Discovery Rate (FDR) level to correct for multiple testing.

    Returns
    -------
    results_df : pandas.DataFrame
        A DataFrame containing the model scores for each gene, along with the
        adjusted p-values and significance based on the significance threshold
        and FDR level.
    """
    # Create initial results dataframe
    results_df = gam_results['scores'].copy()
    results_df['gene'] = results_df.index
    results_df['significant'] = results_df['p_value'] < significance_threshold

    # Keep only non NaN values
    nan_results = results_df.loc[results_df.isna().any(axis=1)]
    results_df = results_df.dropna()

    # Calculate FDR using statsmodels
    _, adj_pvals, _, _ = multipletests(
        results_df['p_value'],
        alpha=fdr_level,
        method='fdr_bh'  # Benjamini-Hochberg FDR
    )
    # Add adjusted p-values and significance
    results_df['adj_p_value'] = adj_pvals
    nan_results['adj_p_value'] = np.nan

    # Add back NaN results
    results_df = pd.concat([results_df, nan_results], axis=0)
    results_df['significant_fdr'] = results_df['adj_p_value'] < fdr_level

    # Sort by explained deviance for final output
    return results_df.sort_values('explained_deviance', ascending=False)