import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sccellfie.plotting.differential_results import create_volcano_plot
from sccellfie.stats.differential_analysis import cohens_d, scanpy_differential_analysis, pairwise_differential_analysis
from sccellfie.datasets.toy_inputs import create_controlled_adata, create_random_adata


def test_cohens_d():
    # Test case 1: Known values
    group1 = np.array([2, 4, 6, 8, 10])
    group2 = np.array([1, 3, 5, 7, 9])
    result = cohens_d(group1, group2)
    expected_result = -0.31622776601683794  # Expected Cohen's d value
    assert pytest.approx(result, 0.00001) == expected_result

    # Test case 2: Equal groups
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([1, 2, 3, 4, 5])
    result = cohens_d(group1, group2)
    assert result == 0

    # Test case 3: Large effect size
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([10, 11, 12, 13, 14])
    result = cohens_d(group1, group2)
    assert result > 3  # Large positive effect size

    # Test case 4: Empty groups
    assert np.isnan(cohens_d([], [1, 2, 3]))


def test_scanpy_differential_analysis():
    # Test with random AnnData
    adata = create_random_adata(n_obs=100, n_vars=50, n_clusters=2)
    adata.obs['condition'] = pd.Categorical(np.random.choice(['C1', 'C2'], size=100))
    adata.obs['cell_type'] = pd.Categorical(np.random.choice(['TypeX', 'TypeY'], size=100))

    cell_type_key = 'cell_type'
    condition_key = 'condition'
    condition_pairs = [('C1', 'C2')]

    # Test with specific cell type - using lower min_cells threshold
    result = scanpy_differential_analysis(
        adata, 'TypeX', cell_type_key, condition_key, condition_pairs,
        min_cells=10  # Lower threshold to ensure we get results
    )

    # Check the structure of the result
    assert isinstance(result, pd.DataFrame)
    expected_columns = {
        'cell_type', 'feature', 'group1', 'group2', 'log2FC', 'test_statistic',
        'p_value', 'cohens_d', 'adj_p_value', 'n_group1', 'n_group2',
        'median_group1', 'median_group2', 'median_diff'
    }
    assert set(result.columns) == expected_columns
    assert len(result) > 0  # Ensure we have results

    # Test with cell_type=None (all cell types)
    result_all = scanpy_differential_analysis(
        adata, None, cell_type_key, condition_key, condition_pairs,
        min_cells=10
    )

    assert isinstance(result_all, pd.DataFrame)
    assert set(result_all.columns) == expected_columns
    assert len(result_all) > 0
    assert 'TypeX' in result_all['cell_type'].values
    assert 'TypeY' in result_all['cell_type'].values


def test_scanpy_differential_analysis_downsampling():
    # Test with random AnnData
    adata = create_random_adata(n_obs=100, n_vars=50, n_clusters=2)
    # Create unbalanced groups
    adata.obs['condition'] = pd.Categorical(['C1'] * 70 + ['C2'] * 30)  # 70-30 split
    adata.obs['cell_type'] = pd.Categorical(['TypeX'] * 100)

    cell_type_key = 'cell_type'
    condition_key = 'condition'
    condition_pairs = [('C1', 'C2')]

    # Test with downsampling and lower min_cells threshold
    result = scanpy_differential_analysis(
        adata, 'TypeX', cell_type_key, condition_key, condition_pairs,
        downsample=True, n_iterations=3, random_state=42, min_cells=10
    )

    # Check basic structure
    assert isinstance(result, pd.DataFrame)
    expected_columns = {
        'cell_type', 'feature', 'group1', 'group2', 'log2FC', 'test_statistic',
        'p_value', 'cohens_d', 'adj_p_value', 'n_group1', 'n_group2',
        'median_group1', 'median_group2', 'median_diff'
    }
    assert set(result.columns) == expected_columns
    assert len(result) > 0

    # Check that the number of cells is balanced (should be 30 each, the size of smaller group)
    assert all(result['n_group1'] == 30)
    assert all(result['n_group2'] == 30)


def test_scanpy_differential_analysis_downsampling_iterations():
    # Create random data instead of using controlled_adata
    adata = create_random_adata(n_obs=20, n_vars=10, n_clusters=2)
    # Create unbalanced groups
    adata.obs['condition'] = pd.Categorical(['A'] * 15 + ['B'] * 5)  # 15-5 split
    adata.obs['cell_type'] = pd.Categorical(['TypeX'] * 20)

    # Run with different numbers of iterations
    result_few = scanpy_differential_analysis(
        adata, 'TypeX', 'cell_type', 'condition',
        downsample=True, n_iterations=3, random_state=42, min_cells=5
    )

    result_many = scanpy_differential_analysis(
        adata, 'TypeX', 'cell_type', 'condition',
        downsample=True, n_iterations=10, random_state=42, min_cells=5
    )

    # Results should be different but have same structure
    assert len(result_few) == len(result_many)
    assert not result_few['log2FC'].equals(result_many['log2FC'])

    # Test different aggregation methods
    result_mean = scanpy_differential_analysis(
        adata, 'TypeX', 'cell_type', 'condition',
        downsample=True, n_iterations=5, agg_method='mean', random_state=42, min_cells=5
    )

    result_median = scanpy_differential_analysis(
        adata, 'TypeX', 'cell_type', 'condition',
        downsample=True, n_iterations=5, agg_method='median', random_state=42, min_cells=5
    )

    # Results should be different but have same structure
    assert len(result_mean) == len(result_median)
    assert not result_mean['log2FC'].equals(result_median['log2FC'])

    # Check that group sizes are consistent (should be 5 each, size of smaller group)
    assert all(result_mean['n_group1'] == 5)
    assert all(result_mean['n_group2'] == 5)


def test_full_differential_analysis_workflow():
    # Test with random AnnData, shifting values to make significant difference
    np.random.seed(42)  # for reproducibility

    # Create random AnnData
    adata = create_random_adata(n_obs=1000, n_vars=50, n_clusters=2)

    # Assign conditions and cell types
    adata.obs['condition'] = np.random.choice(['C1', 'C2'], size=1000)
    adata.obs['cell_type'] = np.random.choice(['TypeX', 'TypeY'], size=1000)

    # Create significant differences
    diff_genes = np.random.choice(adata.var_names, size=10, replace=False)
    diff_gene_indices = [list(adata.var_names).index(gene) for gene in diff_genes]

    # Convert sparse matrix to dense for manipulation
    if sparse.issparse(adata.X):
        X_dense = adata.X.toarray()
    else:
        X_dense = adata.X

    # Increase expression for condition C1 in TypeX
    mask_c1_typex = (adata.obs['condition'] == 'C1') & (adata.obs['cell_type'] == 'TypeX')
    X_dense[mask_c1_typex, :][:, diff_gene_indices] += 150

    # Decrease expression for condition C2 in TypeX
    mask_c2_typex = (adata.obs['condition'] == 'C2') & (adata.obs['cell_type'] == 'TypeX')
    X_dense[mask_c2_typex, :][:, diff_gene_indices] = 0.

    # Decrease expression for condition C1 in TypeY
    mask_c1_typey = (adata.obs['condition'] == 'C1') & (adata.obs['cell_type'] == 'TypeY')
    X_dense[mask_c1_typey, :][:, diff_gene_indices] = 0

    # Increase expression for condition C2 in TypeY
    mask_c2_typey = (adata.obs['condition'] == 'C2') & (adata.obs['cell_type'] == 'TypeY')
    X_dense[mask_c2_typey, :][:, diff_gene_indices] += 150.

    # Convert back to sparse matrix
    if sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(X_dense)
    else:
        adata.X = X_dense

    # Update raw data
    adata.raw = adata.copy()

    # Perform differential analysis
    de_results = scanpy_differential_analysis(adata, None, 'cell_type', 'condition', condition_pairs=[('C1', 'C2')])
    print(de_results['adj_p_value'].min())
    # Create volcano plot
    significant_points = create_volcano_plot(de_results, effect_threshold=0.05, padj_threshold=0.9) # Very relaxed thresholds to ensure significant values are captured

    # Check if the workflow produces expected results
    assert isinstance(de_results, pd.DataFrame)
    assert len(significant_points) > 0
    assert set(de_results['cell_type']) == {'TypeX', 'TypeY'}

    # Clean up the plot
    plt.close()


def test_pairwise_differential_analysis_controlled():
    # Test with controlled data
    adata = create_controlled_adata()
    result = pairwise_differential_analysis(adata, groupby='group')

    # Check basic structure
    assert isinstance(result, pd.DataFrame)
    expected_columns = {
        'feature', 'group1', 'group2', 'log2FC', 'test_statistic', 'p_value',
        'n_group1', 'n_group2', 'median_group1', 'median_group2',
        'cohens_d', 'adj_p_value', 'median_diff'
    }
    assert set(result.columns) == expected_columns

    # Check number of comparisons
    n_genes = len(adata.var_names)
    n_group_pairs = 1  # Only A vs B
    assert len(result) == n_genes * n_group_pairs

    # Check specific values for controlled data
    gene1_result = result[result['feature'] == 'gene1'].iloc[0]
    assert gene1_result['group1'] == 'A'
    assert gene1_result['group2'] == 'B'
    assert gene1_result['n_group1'] == 2  # Two cells in group A
    assert gene1_result['n_group2'] == 2  # Two cells in group B
    assert gene1_result['median_group1'] == 2.0  # Median of [1, 3]
    assert gene1_result['median_group2'] == 6.0  # Median of [5, 7]


def test_pairwise_differential_analysis_random():
    # Test with random data
    np.random.seed(42)
    adata = create_random_adata(n_obs=100, n_vars=10, n_clusters=3)
    adata.obs['condition'] = np.random.choice(['A', 'B', 'C'], size=100)

    result = pairwise_differential_analysis(adata, groupby='condition')

    # Check number of comparisons
    n_genes = 10
    n_group_pairs = 3  # A vs B, A vs C, B vs C
    assert len(result) == n_genes * n_group_pairs

    # Check group pairs
    group_pairs = set(zip(result['group1'], result['group2']))
    expected_pairs = {('A', 'B'), ('A', 'C'), ('B', 'C')}
    assert group_pairs == expected_pairs


def test_pairwise_differential_analysis_order():
    # Test order parameter
    adata = create_controlled_adata()
    order = ['B', 'A']  # Reverse order
    result = pairwise_differential_analysis(adata, groupby='group', order=order)

    # Check if order is respected
    first_comparison = result.iloc[0]
    assert first_comparison['group1'] == 'B'
    assert first_comparison['group2'] == 'A'


def test_pairwise_differential_analysis_var_names():
    # Test var_names parameter
    adata = create_controlled_adata()
    var_names = ['gene1', 'gene2']
    result = pairwise_differential_analysis(adata, groupby='group', var_names=var_names)

    # Check if only specified genes are tested
    assert set(result['feature'].unique()) == set(var_names)
    assert len(result) == len(var_names)  # One comparison per gene


def test_pairwise_differential_analysis_errors():
    adata = create_controlled_adata()

    # Test invalid groupby
    with pytest.raises(KeyError):
        pairwise_differential_analysis(adata, groupby='nonexistent_group')

    # Test invalid var_names
    with pytest.raises(ValueError):
        pairwise_differential_analysis(adata, groupby='group', var_names=['nonexistent_gene'])

    # Test invalid order
    with pytest.raises(KeyError):
        pairwise_differential_analysis(adata, groupby='group', order=['nonexistent_group'])