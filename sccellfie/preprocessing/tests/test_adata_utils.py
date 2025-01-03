import pytest
import numpy as np

from scipy import sparse
from scipy.sparse import issparse
from unittest.mock import patch

from sccellfie.preprocessing.adata_utils import stratified_subsample_adata, normalize_adata, transform_adata_gene_names, transfer_variables
from sccellfie.datasets.toy_inputs import create_random_adata, create_controlled_adata


def test_stratified_subsample_adata():
    # Create a random AnnData object
    n_obs = 1000
    n_vars = 50
    n_clusters = 5
    adata = create_random_adata(n_obs=n_obs, n_vars=n_vars, n_clusters=n_clusters)

    # Perform stratified subsampling
    target_fraction = 0.2
    subsampled_adata = stratified_subsample_adata(adata, group_column='cluster', target_fraction=target_fraction)

    # Check if the subsampled data has approximately the correct size
    expected_size = int(n_obs * target_fraction)
    assert abs(len(subsampled_adata) - expected_size) <= n_clusters  # Allow for small rounding differences

    # Check if all clusters are represented in the subsampled data
    original_clusters = set(adata.obs['cluster'])
    subsampled_clusters = set(subsampled_adata.obs['cluster'])
    assert original_clusters == subsampled_clusters

    # Check if the proportion of each cluster is roughly maintained
    original_proportions = adata.obs['cluster'].value_counts(normalize=True)
    subsampled_proportions = subsampled_adata.obs['cluster'].value_counts(normalize=True)

    for cluster in original_clusters:
        assert np.isclose(original_proportions[cluster], subsampled_proportions[cluster], atol=0.05)


def test_normalize_adata():
    # Create controlled test data
    adata = create_controlled_adata()

    # Add total counts to the adata object
    adata.obs['n_counts'] = np.array([3, 9, 21, 21])

    # Run the preprocessing function
    adata_processed = normalize_adata(adata, target_sum=1000, n_counts_key='n_counts', copy=True)

    # Check that the data are still sparse
    assert sparse.issparse(adata_processed.X)

    # Check that the normalization was performed correctly
    expected_normalized_X = np.array([
        [333.33, 666.67, 0],
        [333.33, 444.44, 222.22],
        [238.10, 285.71, 476.19],
        [333.33, 380.95, 285.71]
    ])

    np.testing.assert_array_almost_equal(adata_processed.X.toarray(), expected_normalized_X, decimal=2)

    # Check that the normalization info was added to uns
    assert 'normalization' in adata_processed.uns
    assert adata_processed.uns['normalization']['method'] == 'total_counts'
    assert adata_processed.uns['normalization']['target_sum'] == 1000
    assert adata_processed.uns['normalization']['n_counts_key'] == 'n_counts'

    # Check that the original data is preserved in .raw
    assert adata_processed.raw is not None
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    if sparse.issparse(adata_processed.raw.X):
        raw_X = adata_processed.raw.X.toarray()
    else:
        raw_X = adata_processed.raw.X
    np.testing.assert_array_equal(raw_X, X)


# Mock data for testing
MOCK_ENSEMBL2SYMBOL = {
    'ENSG00000000001': 'GENE1',
    'ENSG00000000002': 'GENE2',
    'ENSG00000000003': 'GENE3'
}


@patch('sccellfie.preprocessing.adata_utils.retrieve_ensembl2symbol_data')
def test_transform_adata_gene_names(mock_retrieve):
    # Mock the retrieve_ensembl2symbol_data function
    mock_retrieve.return_value = MOCK_ENSEMBL2SYMBOL

    # Load controlled adata
    controlled_adata = create_controlled_adata()
    controlled_adata.var_names = ['ENSG00000000001', 'ENSG00000000002', 'ENSG00000000003']
    original_adata = controlled_adata.copy()

    # Test with default parameters (copy=True, drop_unmapped=False)
    result = transform_adata_gene_names(controlled_adata)
    assert result is not controlled_adata  # Should be a copy
    assert list(result.var_names) == ['GENE1', 'GENE2', 'GENE3']
    assert result.X.shape == (4, 3)
    assert isinstance(result.X, sparse.csr_matrix)
    assert result.raw is not None

    # Test with copy=False
    controlled_adata = original_adata.copy()
    result = transform_adata_gene_names(controlled_adata, copy=False)
    assert result is controlled_adata  # Should be the same object
    assert list(result.var_names) == ['GENE1', 'GENE2', 'GENE3']
    assert np.array_equal(result.X.toarray(), original_adata.X.toarray())

    # Test with drop_unmapped=True (shouldn't change anything in this case)
    controlled_adata = original_adata.copy()
    result = transform_adata_gene_names(controlled_adata, drop_unmapped=True)
    assert list(result.var_names) == ['GENE1', 'GENE2', 'GENE3']
    assert result.X.shape == (4, 3)

    # Test with custom organism
    controlled_adata = original_adata.copy()
    transform_adata_gene_names(controlled_adata, organism='mouse')
    mock_retrieve.assert_called_with(None, 'mouse')

    # Test with custom filename
    controlled_adata = original_adata.copy()
    transform_adata_gene_names(controlled_adata, filename='custom.csv')
    mock_retrieve.assert_called_with('custom.csv', 'human')

    # Test error when no genes are in Ensembl format
    invalid_adata = original_adata.copy()
    invalid_adata.var_names = ['GENE1', 'GENE2', 'GENE3']
    with pytest.raises(ValueError, match="Not all genes are in Ensembl ID format"):
        transform_adata_gene_names(invalid_adata)

    # Test when retrieve_ensembl2symbol_data returns an empty dictionary
    controlled_adata = original_adata.copy()
    mock_retrieve.return_value = {}
    with pytest.raises(ValueError, match="Failed to retrieve Ensembl ID to gene symbol mapping"):
        transform_adata_gene_names(controlled_adata)

    # Test with partial mapping
    controlled_adata = original_adata.copy()
    partial_mapping = {'ENSG00000000001': 'GENE1', 'ENSG00000000002': 'GENE2'}
    mock_retrieve.return_value = partial_mapping
    result = transform_adata_gene_names(controlled_adata)
    assert list(result.var_names) == ['GENE1', 'GENE2', 'ENSG00000000003']

    # Test with partial mapping and drop_unmapped=True
    controlled_adata = original_adata.copy()
    result = transform_adata_gene_names(controlled_adata, drop_unmapped=True)
    assert list(result.var_names) == ['GENE1', 'GENE2']
    assert result.X.shape == (4, 2)


def create_adata_with_var_metadata(n_obs=50, n_vars=20, prefix='gene'):
    """Helper function to create AnnData with var metadata"""
    adata = create_random_adata(n_obs=n_obs, n_vars=n_vars)
    adata.var_names = [f'{prefix}{i}' for i in range(1, n_vars + 1)]

    # Add some metadata columns
    adata.var['feature_type'] = 'gene'
    adata.var['highly_variable'] = False
    return adata


def test_transfer_single_variable():
    """Test transferring a single variable between adatas with different vars"""
    # Create source and target with different variables and metadata
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20, prefix='source_gene')
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15, prefix='target_gene')

    # Transfer single variable
    var_to_transfer = 'source_gene1'
    result = transfer_variables(adata_target, adata_source, var_to_transfer)

    # Assertions
    assert var_to_transfer in result.var_names
    assert result.n_vars == adata_target.n_vars + 1
    assert np.all(result[:, var_to_transfer].X.toarray() == adata_source[:, var_to_transfer].X.toarray())
    assert result.var.loc[var_to_transfer, 'feature_type'] == adata_source.var.loc[var_to_transfer, 'feature_type']
    assert issparse(result.X)


def test_transfer_multiple_variables():
    """Test transferring multiple variables between adatas with different vars"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20, prefix='source_gene')
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15, prefix='target_gene')

    vars_to_transfer = ['source_gene1', 'source_gene2']
    result = transfer_variables(adata_target, adata_source, vars_to_transfer)

    # Assertions
    assert all(var in result.var_names for var in vars_to_transfer)
    assert result.n_vars == adata_target.n_vars + len(vars_to_transfer)
    for var in vars_to_transfer:
        assert np.all(result[:, var].X.toarray() == adata_source[:, var].X.toarray())
        assert result.var.loc[var, 'feature_type'] == adata_source.var.loc[var, 'feature_type']
    assert issparse(result.X)


def test_existing_variables_warning():
    """Test that warning is raised when variables already exist in target"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15)

    # Make some variables overlap
    overlapping_var = 'gene1'
    adata_source.var_names = [f'gene{i}' for i in range(1, 21)]
    adata_target.var_names = [f'gene{i}' for i in range(1, 16)]

    # Try to transfer variables that already exist
    with pytest.warns(UserWarning, match="Variables already exist in target"):
        result = transfer_variables(adata_target, adata_source, [overlapping_var, 'gene20'])

    # Only gene20 should be transferred as gene1 already exists
    assert 'gene20' in result.var_names
    assert result.n_vars == adata_target.n_vars + 1


def test_transfer_with_layers():
    """Test transferring variables when layers are present"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20, prefix='source_gene')
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15, prefix='target_gene')

    # Add layers
    adata_source.layers['counts'] = adata_source.X.copy()
    adata_target.layers['counts'] = adata_target.X.copy()

    vars_to_transfer = ['source_gene1', 'source_gene2']
    result = transfer_variables(adata_target, adata_source, vars_to_transfer)

    # Assertions
    assert 'counts' in result.layers
    assert result.layers['counts'].shape[1] == result.n_vars
    assert issparse(result.X)


def test_transfer_with_obs_mapping():
    """Test transferring variables using observation mapping columns"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20, prefix='source_gene')
    adata_target = create_adata_with_var_metadata(n_obs=40, n_vars=15, prefix='target_gene')

    # Add matching columns
    adata_source.obs['cell_id'] = [f'cell_{i}' for i in range(50)]
    adata_target.obs['cell_id'] = [f'cell_{i}' for i in range(40)]

    result = transfer_variables(
        adata_target,
        adata_source,
        'source_gene1',
        source_obs_col='cell_id',
        target_obs_col='cell_id'
    )

    # Assertions
    assert 'source_gene1' in result.var_names
    assert result.n_obs == adata_target.n_obs
    assert issparse(result.X)


def test_missing_variable():
    """Test that function raises error when requested variable doesn't exist"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15)

    with pytest.raises(ValueError, match="Variables not found in source"):
        transfer_variables(adata_target, adata_source, 'nonexistent_gene')


def test_keep_sparse_false():
    """Test transferring variables with keep_sparse=False"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20, prefix='source_gene')
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15, prefix='target_gene')

    result = transfer_variables(adata_target, adata_source, 'source_gene1', keep_sparse=False)

    # Assertions
    assert not issparse(result.X)
    assert isinstance(result.X, np.ndarray)


def test_var_metadata_transfer():
    """Test that variable metadata is correctly transferred"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20, prefix='source_gene')
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15, prefix='target_gene')

    # Add additional metadata to source
    adata_source.var['new_meta'] = [f'meta{i}' for i in range(1, 21)]

    var_to_transfer = 'source_gene1'
    result = transfer_variables(adata_target, adata_source, var_to_transfer)

    # Assertions
    assert 'feature_type' in result.var.columns
    assert 'new_meta' in result.var.columns
    assert result.var.loc[var_to_transfer, 'new_meta'] == adata_source.var.loc[var_to_transfer, 'new_meta']