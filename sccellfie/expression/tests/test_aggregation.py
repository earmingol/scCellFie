import pytest
import numpy as np
import pandas as pd
import scanpy as sc

from sccellfie.expression.aggregation import agg_expression_cells
from sccellfie.tests.toy_inputs import create_test_adata

def test_agg_expression_cells_all_groups_present():
    adata = create_test_adata()
    groupby = 'group'
    # Create two groups for simplicity
    adata.obs[groupby] = ['group1' if i < adata.n_obs // 2 else 'group2' for i in range(adata.n_obs)]

    # Test aggregation across all groups
    agg_result = agg_expression_cells(adata, groupby, agg_func='mean')

    # Check if all groups are present in the result
    expected_groups = set(adata.obs[groupby].unique())
    result_groups = set(agg_result.index)
    assert expected_groups == result_groups, "Not all groups are present in the aggregation result"


def test_agg_expression_cells_specific_gene_present():
    adata = create_test_adata()
    groupby = 'group'
    adata.obs[groupby] = ['group1' if i < adata.n_obs // 2 else 'group2' for i in range(adata.n_obs)]
    gene_symbols = ['gene1', 'gene10']

    # Test aggregation with specific genes
    agg_result = agg_expression_cells(adata, groupby, gene_symbols=gene_symbols, agg_func='mean')
    assert agg_result.shape == (len(adata.obs[groupby].unique()), len(gene_symbols)), "Shape mismatch"
    assert all(gene in agg_result.columns for gene in gene_symbols), "Missing genes in result"


def test_agg_expression_cells_invalid_agg_func():
    adata = create_test_adata()
    groupby = 'group'
    adata.obs[groupby] = ['group1' if i < adata.n_obs // 2 else 'group2' for i in range(adata.n_obs)]

    # Test with invalid aggregation function
    with pytest.raises(AssertionError):
        agg_expression_cells(adata, groupby, agg_func='invalid_func')


def test_mean_agg_known_values():
    # Create a small, controlled AnnData object
    data = np.array([
        [1, 2],  # Cell 1
        [3, 4],  # Cell 2
        [5, 6],  # Cell 3
    ])
    # Create an AnnData object with known values
    adata = sc.AnnData(X=data)
    adata.var_names = ['gene1', 'gene2']
    adata.obs['group'] = ['A', 'A', 'B']  # Two groups: A and B

    # Expected aggregated values
    # For 'gene1': group A mean = (1+3)/2, group B mean = 5
    # For 'gene2': group A mean = (2+4)/2, group B mean = 6
    expected_means = pd.DataFrame({'gene1': [2, 5], 'gene2': [3, 6]}, index=['A', 'B'], dtype=float)

    # Compute aggregated expression
    agg_result = agg_expression_cells(adata, groupby='group', agg_func='mean')

    # Check if the results match the expected means
    assert agg_result.equals(expected_means), "Aggregated values do not match expected results"

def test_median_agg_known_values():
    # Create a small, controlled AnnData object
    data = np.array([
        [1, 2],  # Cell 1
        [3, 4],  # Cell 2
        [5, 6],  # Cell 3
        [7, 8],  # Cell 4
    ])
    adata = sc.AnnData(X=data)
    adata.var_names = ['gene1', 'gene2']
    adata.obs['group'] = ['A', 'A', 'B', 'B']

    # Expected median values
    # Group A: median for gene1 is (1+3)/2 = 2, for gene2 is (2+4)/2 = 3
    # Group B: median for gene1 is (5+7)/2 = 6, for gene2 is (6+8)/2 = 7
    expected_medians = pd.DataFrame({
        'gene1': [2, 6],
        'gene2': [3, 7]
    }, index=['A', 'B'])

    # Compute aggregated expression using the function
    agg_result = agg_expression_cells(adata, 'group', agg_func='median')

    # Compare the results with the expected values
    assert agg_result.shape == expected_medians.shape, "Shape mismatch"
    assert np.allclose(agg_result.values, expected_medians.values), "Median values do not match expected results"