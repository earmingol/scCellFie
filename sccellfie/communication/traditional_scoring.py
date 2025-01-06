import numpy as np
import pandas as pd

from sccellfie.expression.aggregation import agg_expression_cells


def compute_communication_scores(adata, groupby, var_pairs, communication_score='gmean', agg_func='mean',
                                 layer=None):
    """
    Computes communication scores between pairs of features or variables
    (normally representing ligand-receptor pairs).

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data and grouping information

    groupby : str
        Column in adata.obs for grouping cells to aggregate expression.

    var_pairs : list of tuples
        List of (var1, var2) pairs (normally representing ligand-receptor pairs).

    agg_func : str
        Aggregation function for aggregating expression values across cells.
        Options are 'mean', 'median', '25p' (25th percentile), '75p' (75th percentile),
        'trimean' (0.5*Q2 + 0.25(Q1+Q3)), and 'topmean'
        (computed among the top `top_percent`% of values).

    layer : str, optional
        Layer in adata to use for aggregation. If None, the main expression matrix adata.X is used.

    Returns
    -------
    ccc_scores : pandas.DataFrame
        DataFrame containing the communication score from the expression values for each variable pair
        (ligand-receptor pair).
    """
    # Split variable pairs
    vars1, vars2 = zip(*var_pairs)

    # Check if variables are present in adata
    missing_vars = list(set([var for var in vars1 + vars2 if var not in adata.var_names]))
    if missing_vars:
        raise ValueError(f'Variables not found in adata.var_names: {missing_vars}')

    # Aggregate expression
    agg_df = agg_expression_cells(adata, groupby, layer=layer, agg_func=agg_func)

    # Filter for both gene lists
    df1 = agg_df[list(vars1)]
    df2 = agg_df[list(vars2)]

    # Element-wise multiplication and square root
    scores = CCC_FUNC[communication_score](df1.values, df2.values)

    # Create result dataframe
    ccc_scores = pd.DataFrame(scores,
                              index=agg_df.index,
                              columns=[f'{var1}^{var2}' for var1, var2 in var_pairs])
    return ccc_scores


CCC_FUNC = {'gmean': lambda x, y: np.sqrt(x * y),
            'product' : lambda x, y: x * y,
            'mean' : lambda x, y: (x + y) / 2,
            }