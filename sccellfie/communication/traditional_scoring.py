import numpy as np
import pandas as pd
from itertools import product

from sccellfie.expression.aggregation import agg_expression_cells


def compute_communication_scores(adata, groupby, var_pairs, communication_score='gmean', agg_func='mean',
                                 layer=None):
    """
    Computes communication scores between pairs of features or variables
    (normally representing ligand-receptor pairs) across different cell types.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data and grouping information

    groupby : str
        Column in adata.obs for grouping cells to aggregate expression.

    var_pairs : list of tuples
        List of (var1, var2) pairs (normally representing ligand-receptor pairs).

    communication_score : str, default='gmean'
        Method to compute communication scores. Options are:
        - 'gmean': geometric mean (sqrt(x * y))
        - 'product': simple multiplication (x * y)
        - 'mean': arithmetic mean ((x + y) / 2)

    agg_func : str, default='mean'
        Aggregation function for aggregating expression values across cells.
        Options are 'mean', 'median', '25p' (25th percentile), '75p' (75th percentile),
        'trimean' (0.5*Q2 + 0.25(Q1+Q3)), and 'topmean'.

    layer : str, optional
        Layer in adata to use for aggregation. If None, the main expression matrix adata.X is used.

    Returns
    -------
    ccc_scores : pandas.DataFrame
        DataFrame containing the communication scores between cell types for each variable pair.
        Index is a MultiIndex with (sender_celltype, receiver_celltype).
        Columns represent the variable pairs (ligand-receptor pairs).
    """
    # Split variable pairs
    vars1, vars2 = zip(*var_pairs)

    # Check if variables are present in adata
    missing_vars = list(set([var for var in vars1 + vars2 if var not in adata.var_names]))
    if missing_vars:
        raise ValueError(f'Variables not found in adata.var_names: {missing_vars}')

    # Aggregate expression
    agg_df = agg_expression_cells(adata, groupby, layer=layer, agg_func=agg_func)
    cell_types = agg_df.index.unique()

    # Initialize results dictionary
    results = []

    # Calculate scores for each combination of cell types
    for sender, receiver in product(cell_types, cell_types):
        # Get expression values for sender (vars1/ligands) and receiver (vars2/receptors)
        sender_expr = agg_df.loc[sender, list(vars1)].values
        receiver_expr = agg_df.loc[receiver, list(vars2)].values

        # Calculate communication scores
        scores = CCC_FUNC[communication_score](sender_expr, receiver_expr)

        # Store results
        results.append({
            'sender': sender,
            'receiver': receiver,
            **{f'{var1}^{var2}': score for (var1, var2), score in zip(var_pairs, scores)}
        })

    # Create result dataframe
    ccc_scores = pd.DataFrame(results)
    ccc_scores.set_index(['sender', 'receiver'], inplace=True)

    return ccc_scores


CCC_FUNC = {'gmean': lambda x, y: np.sqrt(x * y),
            'product' : lambda x, y: x * y,
            'mean' : lambda x, y: (x + y) / 2,
            }