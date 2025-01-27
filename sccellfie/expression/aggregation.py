import numpy as np
import pandas as pd
from scipy.sparse import issparse


def agg_expression_cells(adata, groupby, layer=None, gene_symbols=None, agg_func='mean', top_percent=10,
                         exclude_zeros=False, use_raw=False, threshold=None):
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
        'fraction_above' (fraction of cells above threshold)
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

    Notes
    -----
    This function is used to compute summary statistics of gene expression data
    across different groups of cells. It is useful for exploring expression
    patterns in different cell types or conditions.

    The function relies on the `groupby` parameter in `adata.obs` to define the
    groups of cells for which the expression data will be aggregated.
    """
    # Check if agg_func is fraction_above and threshold is provided
    if agg_func == 'fraction_above' and threshold is None:
        raise ValueError("Must provide threshold when using 'fraction_above' aggregation")

    assert agg_func in AGG_FUNC.keys(), "Specify a valid `agg_func`."

    # Select the appropriate data layer
    if layer is not None:
        X = adata.layers[layer]
    else:
        if use_raw:
            X = adata.raw.X
        else:
            X = adata.X

    if issparse(X):
        X = X.toarray()

    if exclude_zeros:
        X = np.where(X==0, np.nan, X)

    # Filter for specific genes if provided
    if gene_symbols is not None:
        if isinstance(gene_symbols, str):
            gene_symbols = [gene_symbols]
        gene_mask = adata.var_names.isin(gene_symbols)
        X = X[:, gene_mask]
        gene_index = adata.var_names[gene_mask]
    else:
        gene_index = adata.var_names

    # Group data by the specified groupby column
    grouped = adata.obs[groupby]
    agg_expression = pd.DataFrame(index=gene_index)

    # Perform aggregation for each group
    for group in sorted(np.unique(grouped)):
        group_mask = grouped == group
        group_data = X[group_mask, :]

        if agg_func == 'topmean':
            agg_expression[group] = AGG_FUNC[agg_func](group_data, axis=0, percent=top_percent)
        elif agg_func == 'fraction_above':
            agg_expression[group] = AGG_FUNC[agg_func](group_data, axis=0, threshold=threshold)
        else:
            agg_expression[group] = AGG_FUNC[agg_func](group_data, axis=0)

    return agg_expression.transpose()


def top_mean(x, axis, percent=10):
    """
    Computes the mean of the top x% values along the specified axis of a matrix, handling NaN values.

    Parameters
    ----------
    x : numpy.ndarray
        The input matrix containing the data to be aggregated.

    axis : int
        The axis along which to compute the mean. Use 0 for columns, 1 for rows.

    percent : float, (default: 10)
        The percentage of top values to consider, ranging from 0 to 100.
        For example, 10 would compute the mean of the top 10% of values.

    Returns
    -------
    numpy.ndarray
        An array containing the mean of the top x% values for each row or column,
        depending on the specified axis. The shape of the output array will be
        (n_rows,) if axis=1, or (n_columns,) if axis=0.
    """

    def top_nanmean(arr, p):
        # Ensure the input is a numpy array
        arr = np.asarray(arr)

        # Remove NaN values
        non_nan = arr[~np.isnan(arr)]

        # If all values are NaN, return NaN
        if len(non_nan) == 0:
            return np.nan

        # Calculate the number of elements to keep
        n = max(1, int(np.ceil(len(arr) * p / 100)))

        # Sort non-NaN values and select top n
        sorted_arr = np.sort(non_nan)
        top_values = sorted_arr[-n:]

        # Return the mean of the top values
        return np.mean(top_values)

    # Apply the function along the specified axis
    return np.apply_along_axis(top_nanmean, axis, x, percent)


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


AGG_FUNC = {'mean' : np.nanmean,
            'median' : np.nanmedian,
            '25p' : lambda x, axis: np.nanpercentile(x, q=25, axis=axis),
            '75p' : lambda x, axis: np.nanpercentile(x, q=75, axis=axis),
            'trimean' : lambda x, axis: 0.5*np.nanpercentile(x, q=50, axis=axis) + 0.25*(np.nanpercentile(x, q=25, axis=axis) + np.nanpercentile(x, q=75, axis=axis)),
            'topmean' : top_mean,
            'fraction_above' : fraction_above_threshold
            }