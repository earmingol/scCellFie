import pytest
import scanpy as sc
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.sparse import issparse, csr_matrix
from unittest.mock import patch

from sccellfie.preprocessing.adata_utils import get_adata_gene_expression, stratified_subsample_adata, normalize_adata, transform_adata_gene_names, transfer_variables
from sccellfie.datasets.toy_inputs import create_random_adata, create_controlled_adata


# Stratifed subsampling tests
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
    expected_size = round(n_obs * target_fraction)
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


# Normalization tests
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


def test_normalize_adata_without_ncounts():
    adata = create_controlled_adata()
    with pytest.warns(UserWarning, match="n_counts not found in adata.obs"):
        normalize_adata(adata, target_sum=1000)


def test_normalize_adata_dense():
    """Test normalization with dense matrix"""
    adata = create_controlled_adata()
    adata.X = adata.X.toarray()

    with pytest.warns(UserWarning, match="Normalizing data"):
        normalize_adata(adata, target_sum=1000)


# Transform gene names tests
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


# Transfer variables tests
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


def test_transfer_variables_missing_layers():
    """Test behavior when source and target have different layers"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15)

    var_to_transfer = adata_source.var_names[0]  # Get an actual var name
    adata_target.layers['test_layer'] = adata_target.X.copy()

    result = transfer_variables(adata_target, adata_source, var_to_transfer)
    assert 'test_layer' in result.layers


def test_transfer_variables_obsm():
    """Test that obsm is properly copied"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15)

    var_to_transfer = adata_source.var_names[0]
    adata_target.obsm['test_obsm'] = np.random.rand(50, 10)

    result = transfer_variables(adata_target, adata_source, var_to_transfer)
    assert 'test_obsm' in result.obsm


def test_transfer_variables_invalid_mapping():
    """Test error when observation mapping fails"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=40, n_vars=15)

    var_to_transfer = adata_source.var_names[0]
    adata_source.obs['cell_id'] = [f'source_{i}' for i in range(50)]
    adata_target.obs['cell_id'] = [f'target_{i}' for i in range(40)]

    with pytest.raises(ValueError, match="Some observations in target could not be mapped"):
        transfer_variables(adata_target, adata_source, var_to_transfer,
                           source_obs_col='cell_id', target_obs_col='cell_id')


def test_transfer_variables_obs_size_mismatch():
    """Test error when observation counts don't match without mapping columns"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=40, n_vars=15)

    var_to_transfer = adata_source.var_names[0]

    with pytest.raises(ValueError, match="Number of observations don't match"):
        transfer_variables(adata_target, adata_source, var_to_transfer)


# Addition transfer tests
def test_empty_var_columns():
    """Test handling when there are no overlapping var columns"""
    # Create source AnnData with much simpler structure
    source_X = csr_matrix(np.ones((50, 1)) * 5)  # Single variable with value 5
    source_var = pd.DataFrame(
        {'feature_type': ['gene'], 'highly_variable': [False]},
        index=['gene1']
    )
    adata_source = sc.AnnData(
        X=source_X,
        var=source_var,
        obs=pd.DataFrame(index=[f'cell{i}' for i in range(50)])
    )

    # Create target with empty var columns
    target_X = csr_matrix(np.random.rand(50, 15))
    target_var = pd.DataFrame(index=[f'target_gene{i}' for i in range(15)])
    adata_target = sc.AnnData(
        X=target_X,
        var=target_var,
        obs=pd.DataFrame(index=[f'cell{i}' for i in range(50)])
    )
    adata_target.var = adata_target.var[[]]  # Remove all columns

    # Transfer the variable
    result = transfer_variables(adata_target, adata_source, 'gene1')

    # Verify transfer
    assert 'gene1' in result.var_names
    assert np.allclose(result[:, 'gene1'].X.toarray(), 5)


def test_source_obsm_transfer():
    """Test transfer of obsm data from source to target"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15)

    # Add different obsm to source and target
    adata_source.obsm['X_pca'] = np.random.rand(50, 10)
    adata_target.obsm['X_umap'] = np.random.rand(50, 2)

    var_to_transfer = adata_source.var_names[0]
    result = transfer_variables(adata_target, adata_source, var_to_transfer)

    # Check that both obsm are present
    assert 'X_umap' in result.obsm
    assert 'X_pca' in result.obsm
    assert np.array_equal(result.obsm['X_pca'], adata_source.obsm['X_pca'])
    assert np.array_equal(result.obsm['X_umap'], adata_target.obsm['X_umap'])


def test_source_obsp_transfer():
    """Test transfer of obsp data from source to target"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15)

    # Add different obsp to source and target
    adata_source.obsp['connectivities'] = csr_matrix((50, 50))
    adata_target.obsp['distances'] = csr_matrix((50, 50))

    var_to_transfer = adata_source.var_names[0]
    result = transfer_variables(adata_target, adata_source, var_to_transfer)

    # Check that both obsp are present
    assert 'distances' in result.obsp
    assert 'connectivities' in result.obsp
    assert issparse(result.obsp['connectivities'])
    assert issparse(result.obsp['distances'])


def test_categorical_dtype_preservation():
    """Test preservation of categorical dtypes and their ordering in obs"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=50, n_vars=15)

    # Add categorical column with specific ordering
    categories = ['A', 'B', 'C']
    adata_target.obs['cat_col'] = pd.Categorical(
        np.random.choice(categories, size=50),
        categories=categories,
        ordered=True
    )

    var_to_transfer = adata_source.var_names[0]
    result = transfer_variables(adata_target, adata_source, var_to_transfer)

    # Check categorical dtype and ordering is preserved
    assert pd.api.types.is_categorical_dtype(result.obs['cat_col'])
    assert result.obs['cat_col'].cat.ordered
    assert list(result.obs['cat_col'].cat.categories) == categories


def test_source_obsm_transfer_with_mapping():
    """Test transfer of obsm data when using observation mapping"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=40, n_vars=15)

    # Add matching columns and obsm data
    adata_source.obs['cell_id'] = [f'cell_{i}' for i in range(50)]
    adata_target.obs['cell_id'] = [f'cell_{i}' for i in range(40)]
    adata_source.obsm['X_pca'] = np.random.rand(50, 10)

    result = transfer_variables(
        adata_target,
        adata_source,
        adata_source.var_names[0],
        source_obs_col='cell_id',
        target_obs_col='cell_id'
    )

    # Check that obsm data was properly aligned
    assert 'X_pca' in result.obsm
    assert result.obsm['X_pca'].shape[0] == 40  # Should match target obs count
    assert np.array_equal(
        result.obsm['X_pca'],
        adata_source.obsm['X_pca'][:40]  # First 40 cells due to our mapping
    )


def test_source_obsp_transfer_with_mapping():
    """Test transfer of obsp data when using observation mapping"""
    adata_source = create_adata_with_var_metadata(n_obs=50, n_vars=20)
    adata_target = create_adata_with_var_metadata(n_obs=40, n_vars=15)

    # Add matching columns and obsp data
    adata_source.obs['cell_id'] = [f'cell_{i}' for i in range(50)]
    adata_target.obs['cell_id'] = [f'cell_{i}' for i in range(40)]
    adata_source.obsp['connectivities'] = csr_matrix(np.random.rand(50, 50))

    result = transfer_variables(
        adata_target,
        adata_source,
        adata_source.var_names[0],
        source_obs_col='cell_id',
        target_obs_col='cell_id'
    )

    # Check that obsp data was properly aligned
    assert 'connectivities' in result.obsp
    assert result.obsp['connectivities'].shape == (40, 40)  # Should match target obs count
    assert issparse(result.obsp['connectivities'])


# Get adata gene expression tests
@pytest.fixture
def adata():
    return create_controlled_adata()


def test_get_adata_gene_expression_basic(adata):
    expression = get_adata_gene_expression(adata, 'gene1')
    assert np.allclose(expression, [1, 3, 5, 7])


def test_get_adata_gene_expression_layer(adata):
    adata.layers['test'] = csr_matrix([[10, 20, 30],
                                      [40, 50, 60],
                                      [70, 80, 90],
                                      [100, 110, 120]])
    expression = get_adata_gene_expression(adata, 'gene1', layer='test')
    assert np.allclose(expression, [10, 40, 70, 100])


def test_get_adata_gene_expression_raw(adata):
    # Use raw data that's already set during creation
    expression = get_adata_gene_expression(adata, 'gene1', use_raw=True)
    expected = adata.raw.X[:, 0].toarray().flatten()
    assert np.allclose(expression, expected)


def test_get_adata_gene_expression_obs(adata):
    adata.obs['test_value'] = [1.5, 2.5, 3.5, 4.5]
    expression = get_adata_gene_expression(adata, 'test_value')
    assert np.allclose(expression, [1.5, 2.5, 3.5, 4.5])


def test_get_adata_gene_expression_non_numeric_obs(adata):
    adata.obs['category'] = ['A', 'B', 'C', 'D']
    with pytest.raises(ValueError, match="Feature 'category' in adata.obs is not numeric"):
        get_adata_gene_expression(adata, 'category')


def test_get_adata_gene_expression_not_found(adata):
    with pytest.raises(ValueError, match="Feature 'nonexistent' not found"):
        get_adata_gene_expression(adata, 'nonexistent')


def test_get_adata_gene_expression_layer_priority(adata):
    adata.layers['test'] = csr_matrix([[10, 20, 30],
                                      [40, 50, 60],
                                      [70, 80, 90],
                                      [100, 110, 120]])
    expression = get_adata_gene_expression(adata, 'gene1', layer='test', use_raw=True)
    assert np.allclose(expression, [10, 40, 70, 100])