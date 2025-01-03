import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sccellfie.preprocessing.matrix_utils import min_max_normalization, get_gene_expression


# Tests for min_max_normalization function
def test_min_max_normalization_dataframe():
    # Test with DataFrame input
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    result = min_max_normalization(df)

    # Check if values are normalized between 0 and 1
    assert np.allclose(result['A'], [0.0, 0.5, 1.0])
    assert np.allclose(result['B'], [0.0, 0.5, 1.0])


def test_min_max_normalization_array():
    # Test with numpy array input
    arr = np.array([[1, 4], [2, 5], [3, 6]])
    result = min_max_normalization(arr)

    expected = pd.DataFrame({
        0: [0.0, 0.5, 1.0],
        1: [0.0, 0.5, 1.0]
    })
    assert np.allclose(result.values, expected.values)


def test_min_max_normalization_row_wise():
    # Test normalization along rows (axis=1)
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    result = min_max_normalization(df, axis=1)

    # For each row, min should be 0 and max should be 1
    assert np.allclose(result.min(axis=1), [0, 0, 0])
    assert np.allclose(result.max(axis=1), [1, 1, 1])


def test_min_max_normalization_with_nan():
    # Test handling of NaN values
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan]
    })
    result = min_max_normalization(df)

    # NaN values should be filled with 0
    assert not result.isna().any().any()
    assert np.allclose(result.fillna(0).values, result.values)


def test_min_max_normalization_constant_column():
    # Test with constant values (max = min)
    df = pd.DataFrame({
        'A': [2, 2, 2],
        'B': [1, 4, 7]
    })
    result = min_max_normalization(df)

    # Constant column should be filled with 0
    assert np.allclose(result['A'], [0, 0, 0])
    assert np.allclose(result['B'], [0, 0.5, 1.0])


# Tests for get_gene_expression function
def test_get_gene_expression_numpy():
    # Test with numpy array
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    var_names = ['gene1', 'gene2', 'gene3']

    expression = get_gene_expression(matrix, var_names, 'gene2')
    assert np.allclose(expression, [2, 5])


def test_get_gene_expression_sparse():
    # Test with sparse matrix
    matrix = csr_matrix([[1, 2, 3], [4, 5, 6]])
    var_names = ['gene1', 'gene2', 'gene3']

    expression = get_gene_expression(matrix, var_names, 'gene2')
    assert np.allclose(expression, [2, 5])


def test_get_gene_expression_dataframe():
    # Test with DataFrame
    matrix = pd.DataFrame({
        'gene1': [1, 4],
        'gene2': [2, 5],
        'gene3': [3, 6]
    })

    expression = get_gene_expression(matrix, matrix.columns, 'gene2')
    assert np.allclose(expression, [2, 5])


def test_get_gene_expression_with_normalization():
    # Test with normalization enabled
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    var_names = ['gene1', 'gene2', 'gene3']

    expression = get_gene_expression(matrix, var_names, 'gene2', normalize=True)
    assert np.allclose(expression.tolist(), [[0.0], [1.0]])  # Should be normalized between 0 and 1


def test_get_gene_expression_gene_not_found():
    # Test error handling for non-existent gene
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    var_names = ['gene1', 'gene2', 'gene3']

    with pytest.raises(ValueError):
        get_gene_expression(matrix, var_names, 'gene4')


def test_get_gene_expression_invalid_matrix():
    # Test error handling for unsupported matrix type
    matrix = [[1, 2, 3], [4, 5, 6]]  # List of lists is not supported
    var_names = ['gene1', 'gene2', 'gene3']

    with pytest.raises(ValueError):
        get_gene_expression(matrix, var_names, 'gene2')


def test_get_gene_expression_pandas_index():
    # Test with pandas Index for var_names
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    var_names = pd.Index(['gene1', 'gene2', 'gene3'])

    expression = get_gene_expression(matrix, var_names, 'gene2')
    assert np.allclose(expression, [2, 5])