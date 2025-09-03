import pytest
import numpy as np

from sccellfie.datasets.toy_inputs import create_controlled_adata, create_controlled_gpr_dict, create_controlled_task_by_rxn, create_global_threshold
from sccellfie.external.tensor import sccellfie_to_tensor
from sccellfie.gene_score import compute_gene_scores
from sccellfie.reaction_activity import compute_reaction_activity
from sccellfie.metabolic_task import compute_mt_score


@pytest.fixture
def mock_preprocessed_db():
    """Create a mock preprocessed database for testing."""
    # Create base AnnData
    adata = create_controlled_adata()

    # Add sample and celltype information
    adata.obs['sample'] = ['sample1', 'sample1', 'sample2', 'sample2']
    adata.obs['celltype'] = ['typeA', 'typeB', 'typeA', 'typeB']

    # Create scCellFie components
    gpr_dict = create_controlled_gpr_dict()
    task_by_rxn = create_controlled_task_by_rxn()
    thresholds = create_global_threshold(threshold=0.5, n_vars=3)

    # Run scCellFie pipeline components
    compute_gene_scores(adata, thresholds)
    compute_reaction_activity(adata, gpr_dict, use_specificity=True, disable_pbar=True)
    compute_mt_score(adata, task_by_rxn, verbose=False)

    # Create preprocessed_db
    preprocessed_db = {
        'adata': adata,
        'gpr_rules': gpr_dict,
        'task_by_rxn': task_by_rxn,
        'thresholds': thresholds
    }

    return preprocessed_db


def test_sccellfie_to_tensor_basic(mock_preprocessed_db):
    """Test basic functionality of sccellfie_to_tensor."""
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        score_type='metabolic_tasks',
        agg_func='mean',
        verbose=False
    )

    # Check return structure
    assert isinstance(tensor_args, dict)
    required_keys = ['tensor', 'order_names', 'order_labels', 'mask', 'loc_nans']
    for key in required_keys:
        assert key in tensor_args

    # Check tensor dimensions
    tensor = tensor_args['tensor']
    assert tensor.ndim == 3
    assert tensor.shape == (2, 2, 4)  # 2 samples, 2 celltypes, 4 tasks

    # Check order names
    order_names = tensor_args['order_names']
    assert len(order_names) == 3
    assert len(order_names[0]) == 2  # contexts
    assert len(order_names[1]) == 2  # celltypes
    assert len(order_names[2]) == 4  # features


def test_sccellfie_to_tensor_reactions(mock_preprocessed_db):
    """Test using reaction scores instead of metabolic tasks."""
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        score_type='reactions',
        agg_func='mean',
        verbose=False
    )

    # Check tensor dimensions for reactions
    tensor = tensor_args['tensor']
    assert tensor.shape == (2, 2, 4)  # 2 samples, 2 celltypes, 4 reactions

    # Check order labels reflect reactions
    assert 'Reactions' in tensor_args['order_labels'][2]


@pytest.mark.parametrize("agg_func", ['mean', 'median', 'trimean'])
def test_sccellfie_to_tensor_aggregation_methods(mock_preprocessed_db, agg_func):
    """Test different aggregation methods."""
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        score_type='metabolic_tasks',
        agg_func=agg_func,
        verbose=False
    )

    # Should complete without error and have correct structure
    assert tensor_args['tensor'].shape == (2, 2, 4)
    assert not np.all(np.isnan(tensor_args['tensor']))


def test_sccellfie_to_tensor_missing_keys(mock_preprocessed_db):
    """Test error handling for missing keys."""
    # Test missing sample_key
    with pytest.raises(ValueError, match="'missing_sample' not found in adata.obs"):
        sccellfie_to_tensor(
            preprocessed_db=mock_preprocessed_db,
            sample_key='missing_sample',
            celltype_key='celltype',
            verbose=False
        )

    # Test missing celltype_key
    with pytest.raises(ValueError, match="'missing_celltype' not found in adata.obs"):
        sccellfie_to_tensor(
            preprocessed_db=mock_preprocessed_db,
            sample_key='sample',
            celltype_key='missing_celltype',
            verbose=False
        )


def test_sccellfie_to_tensor_invalid_score_type(mock_preprocessed_db):
    """Test error handling for invalid score type."""
    with pytest.raises(ValueError, match="score_type must be either 'metabolic_tasks' or 'reactions'"):
        sccellfie_to_tensor(
            preprocessed_db=mock_preprocessed_db,
            sample_key='sample',
            celltype_key='celltype',
            score_type='invalid_type',
            verbose=False
        )


def test_sccellfie_to_tensor_missing_preprocessed_data():
    """Test error handling for missing preprocessed data."""
    # Test missing adata key
    with pytest.raises(ValueError, match="preprocessed_db must contain 'adata' key"):
        sccellfie_to_tensor(
            preprocessed_db={'wrong_key': 'value'},
            sample_key='sample',
            celltype_key='celltype',
            verbose=False
        )


def test_sccellfie_to_tensor_gene_symbols_filter(mock_preprocessed_db):
    """Test filtering by specific gene symbols."""
    # Test with specific features
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        score_type='metabolic_tasks',
        gene_symbols=['task1', 'task2'],
        verbose=False
    )

    # Should have only 2 features
    assert tensor_args['tensor'].shape[2] == 2
    assert len(tensor_args['order_names'][2]) == 2


def test_sccellfie_to_tensor_custom_order(mock_preprocessed_db):
    """Test custom context ordering."""
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        context_order=['sample2', 'sample1'],
        sort_elements=False,
        verbose=False
    )

    # Check that order is preserved
    contexts = tensor_args['order_names'][0]
    assert contexts == ['sample2', 'sample1']


def test_sccellfie_to_tensor_values_consistency(mock_preprocessed_db):
    """Test that tensor values are consistent with aggregated data."""
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        score_type='metabolic_tasks',
        agg_func='mean',
        verbose=False
    )

    tensor = tensor_args['tensor']

    # Check that we have non-NaN values where expected
    # (specific values depend on the controlled data structure)
    assert not np.all(np.isnan(tensor))

    # Check dimensions match expected structure
    contexts, celltypes, features = tensor_args['order_names']
    assert tensor.shape == (len(contexts), len(celltypes), len(features))


def test_sccellfie_to_tensor_verbose_output(mock_preprocessed_db, capsys):
    """Test verbose output functionality."""
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        score_type='metabolic_tasks',
        agg_func='mean',
        verbose=True
    )

    # Capture printed output
    captured = capsys.readouterr()

    # Check that verbose messages are printed
    assert "Using metabolic_tasks" in captured.out
    assert "Aggregating metabolic_task scores" in captured.out
    assert "Building tensor with dimensions" in captured.out
    assert "Tensor built successfully" in captured.out

    # Verify function still works correctly
    assert tensor_args['tensor'].shape == (2, 2, 4)


def test_sccellfie_to_tensor_missing_attributes():
    """Test error handling when metabolic_tasks/reactions attributes are missing."""
    # Create adata without scCellFie attributes
    adata = create_controlled_adata()
    adata.obs['sample'] = ['sample1', 'sample1', 'sample2', 'sample2']
    adata.obs['celltype'] = ['typeA', 'typeB', 'typeA', 'typeB']

    preprocessed_db = {'adata': adata}

    # Test missing metabolic_tasks
    with pytest.raises(ValueError, match="AnnData object must have 'metabolic_tasks' attribute"):
        sccellfie_to_tensor(
            preprocessed_db=preprocessed_db,
            sample_key='sample',
            celltype_key='celltype',
            score_type='metabolic_tasks',
            verbose=False
        )

    # Test missing reactions
    with pytest.raises(ValueError, match="AnnData object must have 'reactions' attribute"):
        sccellfie_to_tensor(
            preprocessed_db=preprocessed_db,
            sample_key='sample',
            celltype_key='celltype',
            score_type='reactions',
            verbose=False
        )


def test_sccellfie_to_tensor_gene_symbols_edge_cases(mock_preprocessed_db):
    """Test gene_symbols parameter with various edge cases."""
    # Test with single string (not list)
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        score_type='metabolic_tasks',
        gene_symbols='task1',  # Single string
        verbose=False
    )

    # Should have only 1 feature
    assert tensor_args['tensor'].shape[2] == 1
    assert tensor_args['order_names'][2] == ['task1']

    # Test with non-existent gene symbols
    with pytest.raises(ValueError, match="None of the specified gene_symbols found in the data"):
        sccellfie_to_tensor(
            preprocessed_db=mock_preprocessed_db,
            sample_key='sample',
            celltype_key='celltype',
            score_type='metabolic_tasks',
            gene_symbols=['nonexistent_task'],
            verbose=False
        )

    # Test with mix of existing and non-existing gene symbols
    tensor_args = sccellfie_to_tensor(
        preprocessed_db=mock_preprocessed_db,
        sample_key='sample',
        celltype_key='celltype',
        score_type='metabolic_tasks',
        gene_symbols=['task1', 'nonexistent_task', 'task2'],  # Mix of valid/invalid
        verbose=False
    )

    # Should only include existing features
    assert tensor_args['tensor'].shape[2] == 2
    assert set(tensor_args['order_names'][2]) == {'task1', 'task2'}