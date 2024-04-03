import numpy as np
import pandas as pd

from scipy.sparse import issparse


def agg_expression_cells(adata, groupby, layer=None, gene_symbols=None, agg_func='mean', use_raw=False):
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
        '25p' (25th percentile), '75p' (75th percentile), and 'trimean' (0.5*Q2 + 0.25(Q1+Q3)).
        The function must be one of the keys in the `AGG_FUNC` dictionary.

    use_raw : bool, optional  (default: False)
        Whether to use the data in adata.raw.X (True) or in adata.X (False).

    Returns
    -------
    agg_expression : pandas.DataFrame
        A pandas.DataFrame where rows correspond to genes and columns correspond to the
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
        agg_expression[group] = AGG_FUNC[agg_func](group_data, axis=0)

    return agg_expression.transpose()


AGG_FUNC = {'mean' : np.nanmean,
            'median' : np.nanmedian,
            '25p' : lambda x, axis: np.percentile(x, q=25, axis=axis),
            '75p' : lambda x, axis: np.percentile(x, q=75, axis=axis),
            'trimean' : lambda x, axis: 0.5*np.percentile(x, q=50, axis=axis) + 0.25*(np.percentile(x, q=25, axis=axis) + np.percentile(x, q=75, axis=axis))
            }