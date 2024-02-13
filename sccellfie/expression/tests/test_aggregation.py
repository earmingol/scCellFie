import pytest
import numpy as np
import pandas as pd

from sccellfie.expression.aggregation import agg_expression_cells
from sccellfie.tests.toy_inputs import create_random_adata, create_controlled_adata


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