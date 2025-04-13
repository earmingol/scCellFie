import numpy as np
import pandas as pd
from itertools import product

from sccellfie.expression.aggregation import agg_expression_cells


def compute_communication_scores(adata, groupby, var_pairs, communication_score='gmean', agg_func='mean',
                                 layer=None, ligand_threshold=0, receptor_threshold=0):
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

    ligand_threshold : float, default=0
        Threshold for calculating the fraction of cells expressing the ligand.
        Only cells with expression above this threshold are considered as expressing the ligand.

    receptor_threshold : float, default=0
        Threshold for calculating the fraction of cells expressing the receptor.
        Only cells with expression above this threshold are considered as expressing the receptor.

    Returns
    -------
    ccc_scores : pandas.DataFrame
        DataFrame containing the communication scores between cell types for each variable pair.
        Columns are:
            - sender_celltype: type of the sender cell
            - receiver_celltype: type of the receiver cell
            - ligand: name of the ligand
            - receptor: name of the receptor
            - score: communication score
            - ligand_fraction: fraction of sender cells expressing the ligand
            - receptor_fraction: fraction of receiver cells expressing the receptor
    """
    # Split variable pairs
    vars1, vars2 = zip(*var_pairs)

    # Check if variables are present in adata
    missing_vars = list(set([var for var in vars1 + vars2 if var not in adata.var_names]))
    if missing_vars:
        raise ValueError(f'Variables not found in adata.var_names: {missing_vars}')

    # Aggregate expression for scores
    agg_df = agg_expression_cells(adata, groupby, layer=layer, agg_func=agg_func)
    cell_types = agg_df.index.unique()

    # Calculate fraction of cells expressing above threshold
    ligand_fractions = agg_expression_cells(
        adata,
        groupby,
        layer=layer,
        gene_symbols=list(vars1),
        agg_func='fraction_above',
        threshold=ligand_threshold
    )

    receptor_fractions = agg_expression_cells(
        adata,
        groupby,
        layer=layer,
        gene_symbols=list(vars2),
        agg_func='fraction_above',
        threshold=receptor_threshold
    )

    # Initialize results list
    results = []

    # Calculate scores for each combination of cell types and variable pairs
    for sender, receiver in product(cell_types, cell_types):
        for (ligand, receptor) in var_pairs:
            # Get expression values
            ligand_expr = agg_df.loc[sender, ligand]
            receptor_expr = agg_df.loc[receiver, receptor]

            # Calculate communication score
            score = CCC_FUNC[communication_score](ligand_expr, receptor_expr)

            # Create result row
            result_dict = {
                'sender_celltype': sender,
                'receiver_celltype': receiver,
                'ligand': ligand,
                'receptor': receptor,
                'score': score,
                'ligand_fraction': ligand_fractions.loc[sender, ligand],
                'receptor_fraction': receptor_fractions.loc[receiver, receptor]
            }

            results.append(result_dict)

    # Create result dataframe
    ccc_scores = pd.DataFrame(results)

    return ccc_scores


CCC_FUNC = {
    'gmean': lambda x, y: np.sqrt(x * y),
    'product': lambda x, y: x * y,
    'mean': lambda x, y: (x + y) / 2,
}