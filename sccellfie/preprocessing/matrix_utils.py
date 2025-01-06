import numpy as np
import pandas as pd
from scipy.sparse import issparse


def min_max_normalization(df, axis=0):
    """
    Applies min-max normalization along specified axis.

    Parameters
    ----------
    df : pandas.DataFrame or array-like
        The input DataFrame to be normalized.

    axis : int, optional (default: 0)
        The axis along which to normalize. Use 0 to normalize
        each column or 1 to normalize each row using their
        cognate min and max values.

    Returns
    -------
    df_scaled : pandas.DataFrame
        A DataFrame containing the normalized values. Minimum and maximum values
        are calculated along the specified axis. Minimum and maximum values are
        0 and 1, respectively. NaN values are filled with 0.
    """
    if isinstance(df, pd.DataFrame):
        df = df.copy()
    else:
        df = pd.DataFrame(df)
    min_vals = df.min(axis=axis)
    max_vals = df.max(axis=axis)
    df_scaled = df.sub(min_vals, axis=1 - axis).div(max_vals - min_vals, axis=1 - axis).fillna(0)
    return df_scaled


def get_matrix_gene_expression(matrix, var_names, gene, normalize=False):
    """
    Safely extracts expression values for a gene from any matrix type.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix containing the expression data. Rows correspond to cells and columns to genes.

    var_names : list or pandas.Index
        The index or array containing the gene names.

    gene : str
        The gene name to extract.

    normalize : bool, optional (default: False)
        If True, apply min-max normalization to the expression values.

    Returns
    -------
    expression : numpy.ndarray
        An array containing the expression values for the specified gene.
    """
    # Find gene index
    if isinstance(var_names, pd.Index):
        gene_idx = var_names.get_loc(gene)
    elif isinstance(var_names, list):
        gene_idx = var_names.index(gene)
    else:
        gene_idx = np.where(var_names == gene)[0][0]

    # Handle different matrix types
    if issparse(matrix):
        expression = matrix[:, gene_idx].toarray().flatten()
    elif isinstance(matrix, np.ndarray):
        expression = matrix[:, gene_idx].flatten()
    elif isinstance(matrix, pd.DataFrame):
        expression = matrix[gene].values
    else:
        raise ValueError(f"Unsupported matrix type: {type(matrix)}")

    expression = expression.astype(np.float64)

    # Apply normalization if requested
    if normalize:
        expression = min_max_normalization(expression).values

    return expression


def compute_dataframes_correlation(df1, df2, col_name=None, method='spearman'):
    """
    Computes correlations between one column in ´df1´ and all columns in another ´df2´.

    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame of which one column will be correlated against multiple columns in df2.

    df2 : pandas.DataFrame
        DataFrame containing multiple columns to correlate against the single column in df1.

    col_name : str, optional (default: None)
        The name of the column in df1 to correlate against df2. If None, the first column in df1 is used.

    method : str, optional (default: 'spearman')
        The correlation method to use. Either 'pearson' or 'spearman'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with correlation coefficients for each column in multi_column_df
    """
    # Validate correlation method
    if method not in ['pearson', 'spearman']:
        raise ValueError("method must be either 'pearson' or 'spearman'")

    # Validate first DataFrame has only one column
    if len(df1.columns) != 1:
        raise ValueError("First DataFrame should have exactly one column")

    if col_name is not None:
        # Validate column name
        if col_name not in df1.columns:
            raise ValueError(f"Column '{col_name}' not found in df1")
    else:
        # Get the column name from the first DataFrame
        col_name = df1.columns[0]

    # Convert first DataFrame to series
    single_series = df1[col_name]

    # Compute correlations
    correlations = {}
    for column in df2.columns:
        corr = single_series.corr(df2[column], method=method)
        correlations[column] = corr

    # Convert to DataFrame
    correlation_df = pd.DataFrame(correlations.items(),
                                  columns=['Column', col_name])
    correlation_df = correlation_df.set_index('Column')

    return correlation_df.sort_values(by=col_name, ascending=False)