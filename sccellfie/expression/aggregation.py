import numpy as np
import pandas as pd
from scipy.sparse import issparse


def agg_expression_cells(adata, groupby, layer=None, gene_symbols=None, agg_func='mean', top_percent=10,
                         exclude_zeros=False, use_raw=False, threshold=None, dtype=None, chunk_size=None):
    """
    Aggregates gene expression data for specified cell groups in an `AnnData` object.

    Parameters
    ----------
    adata : AnnData
        An `AnnData` object containing the expression data to be aggregated.

    groupby : str
        The key in the `adata.obs` DataFrame to group by. This could be any
        categorical annotation of cells (e.g., cell type, condition).

    layer : str, optional (default: None)
        The name of the layer in `adata` to use for aggregation. If `None`,
        the main expression matrix `adata.X` is used.

    gene_symbols : str or list, optional (default: None)
        Gene names to include in the aggregation. If a string is provided,
        it is converted to a single-element list. If `None`, all genes are included.

    agg_func : str, optional  (default: 'mean')
        The aggregation function to apply. Options are 'mean', 'median',
        '25p' (25th percentile), '75p' (75th percentile), 'trimean' (0.5*Q2 + 0.25(Q1+Q3)),
        'topmean' (computed among the top `top_percent`% of values), and
        'fraction_above' (fraction of cells above threshold).
        The function must be one of the keys in the `AGG_FUNC` dictionary.

    top_percent : float, optional (default: 10)
        The percentage of top values to consider when `agg_func` is 'topmean'.
        Ranging from 0 to 100.

    exclude_zeros: bool, optional (default: False)
        Whether to exclude zeros when aggregating the values.

    use_raw : bool, optional  (default: False)
        Whether to use the data in adata.raw.X (True) or in adata.X (False).

    threshold : float, optional (default: None)
        Expression threshold used when agg_func is 'fraction_above'. Represents the
        minimum expression value for a cell to be considered as expressing the gene.

    dtype : numpy dtype, optional (default: None)
        If provided (e.g. np.float32), dense group blocks are cast to this dtype
        before aggregation to reduce memory. If None, the source dtype is kept.

    chunk_size : int or None, optional (default: None)
        Number of genes to densify at a time when aggregating, to bound peak
        memory. Only affects aggregations that require densification (`median`,
        `25p`, `75p`, `trimean`, `topmean`, and any dense input); the sparse
        `mean` / `fraction_above` fast paths are already streamed. Results are
        unchanged — chunking over genes keeps every cell for each gene, so
        percentiles stay exact. `None` densifies each group in one block.

    Returns
    -------
    agg_expression : pandas.DataFrame
        A pandas.DataFrame where columns correspond to genes and rows correspond to the
        unique categories in `groupby`. Each cell in the DataFrame contains the
        aggregated expression value for the corresponding gene and group.

    Raises
    ------
    AssertionError
        If the provided `agg_func` is not a valid key in `AGG_FUNC`.
    ValueError
        If `agg_func` is 'fraction_above' and no `threshold` is provided.

    Notes
    -----
    This function is used to compute summary statistics of gene expression data
    across different groups of cells. It is useful for exploring expression
    patterns in different cell types or conditions.

    The function relies on the `groupby` parameter in `adata.obs` to define the
    groups of cells for which the expression data will be aggregated.
    """
    # Validate agg_func / threshold combination up front
    if agg_func == 'fraction_above' and threshold is None:
        raise ValueError("Must provide threshold when using 'fraction_above' aggregation")

    assert agg_func in AGG_FUNC.keys(), "Specify a valid `agg_func`."

    # --- Select the appropriate data source (kept sparse if it is sparse) ---
    if layer is not None:
        X = adata.layers[layer]
        var_names = adata.var_names
    elif use_raw:
        X = adata.raw.X
        # raw has its own var_names, which may differ from adata.var_names
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    # --- Subset genes BEFORE densifying so we only ever materialize needed columns ---
    if gene_symbols is not None:
        if isinstance(gene_symbols, str):
            gene_symbols = [gene_symbols]
        gene_mask = var_names.isin(gene_symbols)
        X = X[:, gene_mask]              # sparse column slice stays sparse & cheap
        gene_index = var_names[gene_mask]
    else:
        gene_index = var_names

    grouped = adata.obs[groupby]
    agg_expression = pd.DataFrame(index=gene_index)

    sparse_input = issparse(X)

    # --- Aggregate group by group; only densify what we must ---
    for group in sorted(grouped.unique()):
        group_mask = (grouped == group).values
        group_data = X[group_mask, :]

        # ---- Sparse fast paths (no densification) ----
        if sparse_input and issparse(group_data):
            if agg_func == 'mean' and not exclude_zeros:
                # Mean over all cells, including implicit zeros
                agg_expression[group] = np.asarray(group_data.mean(axis=0)).ravel()
                continue

            if agg_func == 'mean' and exclude_zeros:
                # Mean over nonzero entries only = column_sum / column_nnz
                col_sum = np.asarray(group_data.sum(axis=0)).ravel()
                col_nnz = group_data.getnnz(axis=0)
                with np.errstate(invalid='ignore', divide='ignore'):
                    result = np.where(col_nnz > 0, col_sum / col_nnz, np.nan)
                agg_expression[group] = result
                continue

            if agg_func == 'fraction_above' and threshold >= 0:
                # For threshold >= 0, only stored nonzeros can exceed it, so we can
                # count directly on the sparse matrix without densifying.
                n_cells = group_data.shape[0]
                if n_cells == 0:
                    agg_expression[group] = np.full(group_data.shape[1], np.nan)
                else:
                    above = np.asarray((group_data > threshold).sum(axis=0)).ravel()
                    agg_expression[group] = above / n_cells
                continue

        # ---- Fallback: densify this group, optionally one gene-window at a time ----
        n_cells, n_genes = group_data.shape

        if n_cells == 0:
            agg_expression[group] = np.full(n_genes, np.nan)
            continue

        # Chunk over GENES (never cells) so every cell is seen per gene and
        # percentile/order-statistic results stay exact.
        gene_window = min(chunk_size, n_genes) if chunk_size is not None else n_genes

        col_result = np.empty(n_genes, dtype=float)
        for g_start in range(0, n_genes, gene_window):
            g_end = min(g_start + gene_window, n_genes)
            block = group_data[:, g_start:g_end]

            if issparse(block):
                block = block.toarray()
            else:
                block = np.asarray(block)

            if dtype is not None:
                block = block.astype(dtype, copy=False)

            if exclude_zeros:
                # np.where forces float; do it on the gene-window block only
                block = np.where(block == 0, np.nan, block)

            if agg_func == 'topmean':
                col_result[g_start:g_end] = AGG_FUNC[agg_func](block, axis=0, percent=top_percent)
            elif agg_func == 'fraction_above':
                col_result[g_start:g_end] = AGG_FUNC[agg_func](block, axis=0, threshold=threshold)
            else:
                col_result[g_start:g_end] = AGG_FUNC[agg_func](block, axis=0)

        agg_expression[group] = col_result

    return agg_expression.transpose()


def top_mean(x, axis, percent=10):
    """
    Computes the mean of the top x% values along the specified axis of a matrix,
    handling NaN values.

    Vectorized with np.partition instead of np.apply_along_axis for speed on wide
    matrices.

    Parameters
    ----------
    x : numpy.ndarray
        The input matrix containing the data to be aggregated.

    axis : int
        The axis along which to compute the mean. Use 0 for columns, 1 for rows.

    percent : float, (default: 10)
        The percentage of top values to consider, ranging from 0 to 100.

    Returns
    -------
    numpy.ndarray
        An array containing the mean of the top x% values for each row or column,
        depending on the specified axis.
    """
    x = np.asarray(x, dtype=float)

    # Work along axis 0 internally; transpose if needed so columns are the groups.
    if axis == 1:
        x = x.T
    elif axis != 0:
        raise ValueError("axis must be 0 or 1")

    n_rows, n_cols = x.shape
    result = np.empty(n_cols, dtype=float)

    for j in range(n_cols):
        col = x[:, j]
        non_nan = col[~np.isnan(col)]
        if non_nan.size == 0:
            result[j] = np.nan
            continue
        # Number of top elements to keep: relative to the full column length,
        # matching the original implementation.
        n = max(1, int(np.ceil(n_rows * percent / 100)))
        n = min(n, non_nan.size)
        # Partition so the largest n values sit at the end (unordered), then mean.
        if n >= non_nan.size:
            top_values = non_nan
        else:
            top_values = np.partition(non_nan, -n)[-n:]
        result[j] = np.mean(top_values)

    return result


def fraction_above_threshold(x, axis, threshold=0):
    """
    Computes the fraction of values above a threshold along the specified axis.

    Parameters
    ----------
    x : numpy.ndarray
        The input matrix containing the data.

    axis : int
        The axis along which to compute the fraction. Use 0 for columns, 1 for rows.

    threshold : float, (default: 0)
        The threshold value above which to count values.

    Returns
    -------
    numpy.ndarray
        An array containing the fraction (between 0 and 1) of values above threshold.
    """
    return np.mean(x > threshold, axis=axis)


def _trimean(x, axis):
    q1, q2, q3 = np.nanpercentile(x, [25, 50, 75], axis=axis)
    return 0.5 * q2 + 0.25 * (q1 + q3)


AGG_FUNC = {'mean' : np.nanmean,
            'median' : np.nanmedian,
            '25p' : lambda x, axis: np.nanpercentile(x, q=25, axis=axis),
            '75p' : lambda x, axis: np.nanpercentile(x, q=75, axis=axis),
            'trimean' : _trimean,
            'topmean' : top_mean,
            'fraction_above' : fraction_above_threshold
            }