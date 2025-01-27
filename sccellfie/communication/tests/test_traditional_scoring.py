import pytest
import numpy as np
import pandas as pd

from sccellfie.datasets.toy_inputs import create_controlled_adata
from sccellfie.communication import compute_communication_scores


@pytest.fixture
def adata_fixture():
    return create_controlled_adata()


def test_compute_communication_scores_basic(adata_fixture):
    """Test basic functionality and output structure"""
    var_pairs = [('gene1', 'gene2')]
    scores = compute_communication_scores(adata_fixture, 'group', var_pairs)

    assert isinstance(scores, pd.DataFrame)
    expected_columns = ['sender_celltype', 'receiver_celltype', 'ligand',
                        'receptor', 'score', 'ligand_fraction', 'receptor_fraction']
    assert list(scores.columns) == expected_columns

    # Check that we have all cell type pairs (including self-interactions)
    expected_pairs = {('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')}
    actual_pairs = set(zip(scores['sender_celltype'], scores['receiver_celltype']))
    assert actual_pairs == expected_pairs

    assert not scores.isna().any().any()


def test_compute_communication_scores_values(adata_fixture):
    """Test score computation with different methods"""
    var_pairs = [('gene1', 'gene2')]

    # Test gmean
    scores = compute_communication_scores(adata_fixture, 'group', var_pairs)
    ab_score = scores[(scores['sender_celltype'] == 'A') &
                      (scores['receiver_celltype'] == 'B')]['score'].iloc[0]
    assert np.isclose(ab_score, np.sqrt(2 * 7))  # A->B: gene1 from A (2) with gene2 from B (7)

    # Test product
    scores = compute_communication_scores(adata_fixture, 'group', var_pairs,
                                          communication_score='product')
    ab_score = scores[(scores['sender_celltype'] == 'A') &
                      (scores['receiver_celltype'] == 'B')]['score'].iloc[0]
    assert np.isclose(ab_score, 2 * 7)

    # Test mean
    scores = compute_communication_scores(adata_fixture, 'group', var_pairs,
                                          communication_score='mean')
    ab_score = scores[(scores['sender_celltype'] == 'A') &
                      (scores['receiver_celltype'] == 'B')]['score'].iloc[0]
    assert np.isclose(ab_score, (2 + 7) / 2)


def test_compute_communication_scores_fractions(adata_fixture):
    """Test computation of ligand and receptor fractions"""
    var_pairs = [('gene1', 'gene2')]
    scores = compute_communication_scores(adata_fixture, 'group', var_pairs,
                                          ligand_threshold=2, receptor_threshold=4)

    # For group A:
    # gene1 (ligand) values: [1,3] -> fraction above 2: 0.5
    # gene2 (receptor) values: [2,4] -> fraction above 4: 0
    a_sender = scores[scores['sender_celltype'] == 'A'].iloc[0]
    assert np.isclose(a_sender['ligand_fraction'], 0.5)

    # For group B:
    # gene2 (receptor) values: [6,8] -> fraction above 4: 1.0
    b_receiver = scores[scores['receiver_celltype'] == 'B'].iloc[0]
    assert np.isclose(b_receiver['receptor_fraction'], 1.0)


def test_compute_communication_scores_multiple_pairs(adata_fixture):
    """Test with multiple ligand-receptor pairs"""
    var_pairs = [('gene1', 'gene2'), ('gene2', 'gene3')]
    scores = compute_communication_scores(adata_fixture, 'group', var_pairs)

    # Check unique ligand-receptor pairs
    actual_pairs = set(zip(scores['ligand'], scores['receptor']))
    expected_pairs = {('gene1', 'gene2'), ('gene2', 'gene3')}
    assert actual_pairs == expected_pairs

    # Check all cell type combinations exist for each pair
    for ligand, receptor in var_pairs:
        pair_scores = scores[(scores['ligand'] == ligand) &
                             (scores['receptor'] == receptor)]
        actual_pairs = set(zip(pair_scores['sender_celltype'],
                               pair_scores['receiver_celltype']))
        expected_pairs = {('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')}
        assert actual_pairs == expected_pairs


def test_compute_communication_scores_errors(adata_fixture):
    """Test error handling"""
    # Test non-existent gene
    with pytest.raises(ValueError, match='Variables not found'):
        compute_communication_scores(adata_fixture, 'group',
                                     [('nonexistent_gene', 'gene2')])

    # Test invalid scoring method
    with pytest.raises(KeyError):
        compute_communication_scores(adata_fixture, 'group', [('gene1', 'gene2')],
                                     communication_score='invalid_method')

    # Test invalid grouping column
    with pytest.raises(KeyError):
        compute_communication_scores(adata_fixture, 'nonexistent_group',
                                     [('gene1', 'gene2')])


def test_compute_communication_scores_layer(adata_fixture):
    """Test computation using a specific layer"""
    # Add a test layer with doubled values
    adata_fixture.layers['test'] = adata_fixture.X * 2
    var_pairs = [('gene1', 'gene2')]

    # Compare scores with and without layer
    scores_base = compute_communication_scores(adata_fixture, 'group', var_pairs)
    scores_layer = compute_communication_scores(adata_fixture, 'group', var_pairs,
                                                layer='test')

    # Scores should be doubled (for geometric mean)
    ratio = scores_layer['score'] / scores_base['score']
    assert np.allclose(ratio, 2.0)