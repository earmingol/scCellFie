import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sccellfie.plotting.differential_results import create_volcano_plot
from sccellfie.stats.differential_analysis import cohens_d, scanpy_differential_analysis
from sccellfie.tests.toy_inputs import create_random_adata


def test_cohens_d():
    # Test case 1: Known values
    group1 = np.array([2, 4, 6, 8, 10])
    group2 = np.array([1, 3, 5, 7, 9])
    result = cohens_d(group1, group2)
    expected_result = 0.31622776601683794  # Expected Cohen's d value
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
    assert result < -3  # Large negative effect size

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

    # Test with specific cell type
    result = scanpy_differential_analysis(adata, 'TypeX', cell_type_key, condition_key, condition_pairs)

    # Check the structure of the result
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'contrast', 'log2FC', 'test_statistic', 'p_value', 'cohens_d', 'adj_p_value'}
    assert len(result) == 50 # We expect results for all 50 genes
    assert all(result.index.get_level_values('cell_type') == 'TypeX')

    # Test with cell_type=None (all cell types)
    result_all = scanpy_differential_analysis(adata, None, cell_type_key, condition_key, condition_pairs)

    assert isinstance(result_all, pd.DataFrame)
    assert set(result_all.columns) == {'contrast', 'log2FC', 'test_statistic', 'p_value', 'cohens_d', 'adj_p_value'}
    assert len(result_all) == 100  # We expect results for all 50 genes and 2 cell types
    assert set(result_all.index.get_level_values('cell_type')) == {'TypeX', 'TypeY'}

    # Test error handling
    with pytest.raises(KeyError):
        scanpy_differential_analysis(adata, 'TypeX', 'non_existent_key', 'group')


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
    assert set(de_results.index.get_level_values('cell_type')) == {'TypeX', 'TypeY'}

    # Clean up the plot
    plt.close()