import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sparse

from sccellfie.expression.aggregation import agg_expression_cells, top_mean, fraction_above_threshold
from sccellfie.datasets.toy_inputs import create_random_adata, create_controlled_adata


@pytest.mark.parametrize("use_raw", [False, True])
def test_agg_expression_cells_all_groups_present(use_raw):
    adata = create_random_adata()
    groupby = 'group'
    # Create two groups for simplicity
    adata.obs[groupby] = ['group1' if i < adata.n_obs // 2 else 'group2' for i in range(adata.n_obs)]

    # Test aggregation across all groups
    agg_result = agg_expression_cells(adata, groupby, agg_func='mean', use_raw=use_raw)

    # Check if all groups are present in the result
    expected_groups = set(adata.obs[groupby].unique())
    result_groups = set(agg_result.index)
    assert expected_groups == result_groups, "Not all groups are present in the aggregation result"


@pytest.mark.parametrize("use_raw", [False, True])
def test_agg_expression_cells_specific_gene_present(use_raw):
    adata = create_random_adata()
    groupby = 'group'
    adata.obs[groupby] = ['group1' if i < adata.n_obs // 2 else 'group2' for i in range(adata.n_obs)]
    gene_symbols = ['gene1', 'gene10']

    # Test aggregation with one gene
    agg_result = agg_expression_cells(adata, groupby, gene_symbols=gene_symbols[0], agg_func='mean', use_raw=use_raw)
    assert agg_result.shape == (len(adata.obs[groupby].unique()), 1), "Shape mismatch"
    assert gene_symbols[0] in agg_result.columns, "Missing gene in result"

    # Test aggregation with specific genes
    agg_result = agg_expression_cells(adata, groupby, gene_symbols=gene_symbols, agg_func='mean', use_raw=use_raw)
    assert agg_result.shape == (len(adata.obs[groupby].unique()), len(gene_symbols)), "Shape mismatch"
    assert all(gene in agg_result.columns for gene in gene_symbols), "Missing genes in result"


def test_agg_expression_cells_layer():
    adata = create_random_adata(layers='gene_scores')
    groupby = 'group'
    adata.obs[groupby] = ['group1' if i < adata.n_obs // 2 else 'group2' for i in range(adata.n_obs)]

    # Test aggregation with one gene
    agg_result = agg_expression_cells(adata, groupby, layer='gene_scores', agg_func='mean')
    assert agg_result.shape == (len(adata.obs[groupby].unique()), adata.shape[1]), "Shape mismatch"


def test_agg_expression_cells_invalid_agg_func():
    adata = create_random_adata()
    groupby = 'group'
    adata.obs[groupby] = ['group1' if i < adata.n_obs // 2 else 'group2' for i in range(adata.n_obs)]

    # Test with invalid aggregation function
    with pytest.raises(AssertionError):
        agg_expression_cells(adata, groupby, agg_func='invalid_func')


def test_mean_agg_known_values():
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Expected aggregated values
    # For 'gene1': group A mean = (1+3)/2, group B mean = (5+7)/2
    # For 'gene2': group A mean = (2+4)/2, group B mean = (6+8)/2
    # For 'gene3': group A mean = (0+2)/2, group B mean = (10+6)/2
    expected_means = pd.DataFrame({'gene1': [2, 6],
                                   'gene2': [3, 7],
                                   'gene3': [1, 8]},
                                  index=['A', 'B'],
                                  dtype=float)

    # Compute aggregated expression
    agg_result = agg_expression_cells(adata, groupby='group', agg_func='mean')

    # Check if the results match the expected means
    assert agg_result.equals(expected_means), "Aggregated values do not match expected results"


def test_median_agg_known_values():
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Expected median values
    # Group A: median for gene1 is (1+3)/2 = 2, for gene2 is (2+4)/2 = 3, for gene3 is (0+2)/2
    # Group B: median for gene1 is (5+7)/2 = 6, for gene2 is (6+8)/2 = 7, for gene3 is (6+10)/2
    expected_medians = pd.DataFrame({'gene1': [2, 6],
                                     'gene2': [3, 7],
                                     'gene3': [1, 8]},
                                    index=['A', 'B'])

    # Compute aggregated expression using the function
    agg_result = agg_expression_cells(adata, 'group', agg_func='median')

    # Compare the results with the expected values
    assert agg_result.shape == expected_medians.shape, "Shape mismatch"
    assert np.allclose(agg_result.values, expected_medians.values), "Median values do not match expected results"


def test_X_in_agg_expression():
    adata = create_random_adata(layers='gene_scores')
    groupby = 'group'
    adata.obs[groupby] = ['group1' if i < adata.n_obs // 2 else 'group2' for i in range(adata.n_obs)]

    # Test aggregation with sparse X
    agg_result = agg_expression_cells(adata, groupby, agg_func='mean')

    # Test aggregation with dense X
    adata.X = adata.X.toarray()
    agg_result = agg_expression_cells(adata, groupby, agg_func='mean')


def test_top_mean_basic():
    matrix = np.array([[1, 2, np.nan, 4],
                       [5, 6, 7, 8],
                       [9, 10, np.nan, np.nan]])

    result_columns = top_mean(matrix, axis=0, percent=50)
    expected_columns = np.array([7., 8., 7., 6.])
    assert np.allclose(result_columns, expected_columns, equal_nan=True)


def test_top_mean_all_nan():
    matrix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    result = top_mean(matrix, axis=0, percent=50)
    assert np.isnan(result).all()


def test_top_mean_mixed_nan():
    matrix = np.array([[1, np.nan, 3],
                       [np.nan, np.nan, 6],
                       [7, 8, np.nan]])
    result = top_mean(matrix, axis=0, percent=50)
    expected = np.array([4., 8., 4.5])
    assert np.allclose(result, expected, equal_nan=True)


def test_top_mean_single_value():
    matrix = np.array([[1], [2], [3]])
    result = top_mean(matrix, axis=0, percent=10)
    assert result[0] == 3


def test_top_mean_exclude_zeros():
    matrix = np.array([[0, 2, 0],
                       [3, 0, 2],
                       [5, 6, 0],
                       [0, 8, 6]])
    result = top_mean(matrix, axis=0, percent=50)
    expected = np.array([4., 7., 4.])
    assert np.allclose(result, expected)


def test_top_mean_high_percentage_with_nan():
    matrix = np.array([[1, np.nan, 3],
                       [4, 5, np.nan],
                       [7, 8, 9]])
    result = top_mean(matrix, axis=0, percent=90)
    expected = np.array([4., 6.5, 6.])
    assert np.allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize("percent", [1, 25, 50, 75, 100])
def test_top_mean_various_percentages(percent):
    matrix = np.array([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10]])
    result = top_mean(matrix, axis=1, percent=percent)
    expected = np.array([5, 10]) if percent == 1 else np.array([4.5, 9.5]) if percent == 25 else np.array(
        [4., 9.]) if percent == 50 else np.array([3.5, 8.5]) if percent == 75 else np.array([3., 8.])
    assert np.allclose(result, expected)


def test_top_mean_all_zeros():
    matrix = np.zeros((3, 3))
    result = top_mean(matrix, axis=0, percent=50)
    assert np.allclose(result, np.zeros(3))


def test_top_mean_negative_numbers():
    matrix = np.array([[-1, -2, -3],
                       [-4, -5, -6],
                       [-7, -8, -9]])
    result = top_mean(matrix, axis=0, percent=50)
    expected = np.array([-2.5, -3.5, -4.5])
    assert np.allclose(result, expected)


def test_top_mean_large_matrix():
    np.random.seed(0)
    large_matrix = np.random.rand(1000, 1000)
    result = top_mean(large_matrix, axis=0, percent=10)
    assert result.shape == (1000,)
    assert np.all(result > 0.9)  # The top 10% of random numbers between 0 and 1 should all be > 0.9


def test_agg_expression_cells_topmean():
    adata = create_controlled_adata()

    expected_topmean = pd.DataFrame({'gene1': [3, 7],
                                     'gene2': [4, 8],
                                     'gene3': [2, 10]},
                                    index=['A', 'B'])

    agg_result = agg_expression_cells(adata, 'group', agg_func='topmean')
    assert np.allclose(agg_result.values, expected_topmean.values)


def test_top_mean_use_raw():
    adata = create_controlled_adata()
    adata.X = sparse.csr_matrix(np.array([
        [10, 20, 0],
        [30, 40, 20],
        [50, 60, 100],
        [70, 80, 60]
    ]))

    agg_result = agg_expression_cells(adata, 'group', agg_func='topmean', use_raw=True)

    expected_result = pd.DataFrame({'gene1': [3, 7],
                                    'gene2': [4, 8],
                                    'gene3': [2, 10]},
                                   index=['A', 'B'])

    assert np.allclose(agg_result.values, expected_result.values)


def test_fraction_above():
    adata = create_controlled_adata()
    """Test the new fraction_above aggregation function."""
    result = agg_expression_cells(adata, groupby='group',
                                  agg_func='fraction_above', threshold=3)

    # For group A: gene1 (1,3), gene2 (2,4), gene3 (0,2) -> fractions: [0.0, 0.5, 0.0]
    # For group B: gene1 (5,7), gene2 (6,8), gene3 (10,6) -> fractions: [1.0, 1.0, 1.0]
    expected_A = np.array([0.0, 0.5, 0.0])
    expected_B = np.array([1.0, 1.0, 1.0])

    np.testing.assert_array_almost_equal(result.loc['A'], expected_A)
    np.testing.assert_array_almost_equal(result.loc['B'], expected_B)


def test_fraction_above_error():
    """Test that fraction_above raises error when threshold is not provided."""
    adata = create_controlled_adata()
    with pytest.raises(ValueError, match="Must provide threshold when using 'fraction_above' aggregation"):
        agg_expression_cells(adata, groupby='group', agg_func='fraction_above')


def test_fraction_above_threshold_function():
    """Test the fraction_above_threshold helper function directly."""
    test_data = np.array([[1, 5, 3],
                          [4, 2, 6],
                          [7, 8, 9]])

    # Test along axis 0 (columns)
    result = fraction_above_threshold(test_data, axis=0, threshold=5)
    expected = np.array([1 / 3, 1 / 3, 2 / 3])  # fraction of values > 5 in each column
    np.testing.assert_array_almost_equal(result, expected)

    # Test along axis 1 (rows)
    result = fraction_above_threshold(test_data, axis=1, threshold=5)
    expected = np.array([0.0, 1 / 3, 1.0])  # fraction of values > 5 in each row
    np.testing.assert_array_almost_equal(result, expected)