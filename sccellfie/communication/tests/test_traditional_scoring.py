import pytest
import numpy as np
import pandas as pd

from sccellfie.communication.traditional_scoring import compute_communication_scores
from sccellfie.datasets.toy_inputs import create_controlled_adata


def test_compute_communication_scores_basic():
    """Test basic functionality and output structure"""
    adata = create_controlled_adata()
    var_pairs = [('gene1', 'gene2')]
    scores = compute_communication_scores(adata, 'group', var_pairs)

    assert isinstance(scores, pd.DataFrame)
    assert list(scores.columns) == ['gene1^gene2']
    # Check that we have all cell type pairs (including self-interactions)
    expected_pairs = {('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')}
    assert set(map(tuple, scores.index)) == expected_pairs
    assert not scores.isna().any().any()


def test_compute_communication_scores_values():
    """Test score computation with different methods"""
    adata = create_controlled_adata()
    var_pairs = [('gene1', 'gene2')]

    # Test gmean
    scores = compute_communication_scores(adata, 'group', var_pairs)
    # A->B interaction: gene1 from A (2) with gene2 from B (7)
    assert np.isclose(scores.loc[('A', 'B'), 'gene1^gene2'], np.sqrt(2 * 7))
    # B->A interaction: gene1 from B (6) with gene2 from A (3)
    assert np.isclose(scores.loc[('B', 'A'), 'gene1^gene2'], np.sqrt(6 * 3))
    # A->A interaction: gene1 from A (2) with gene2 from A (3)
    assert np.isclose(scores.loc[('A', 'A'), 'gene1^gene2'], np.sqrt(2 * 3))
    # B->B interaction: gene1 from B (6) with gene2 from B (7)
    assert np.isclose(scores.loc[('B', 'B'), 'gene1^gene2'], np.sqrt(6 * 7))

    # Test product
    scores = compute_communication_scores(adata, 'group', var_pairs, communication_score='product')
    assert np.isclose(scores.loc[('A', 'B'), 'gene1^gene2'], 2 * 7)
    assert np.isclose(scores.loc[('B', 'A'), 'gene1^gene2'], 6 * 3)
    assert np.isclose(scores.loc[('A', 'A'), 'gene1^gene2'], 2 * 3)
    assert np.isclose(scores.loc[('B', 'B'), 'gene1^gene2'], 6 * 7)

    # Test mean
    scores = compute_communication_scores(adata, 'group', var_pairs, communication_score='mean')
    assert np.isclose(scores.loc[('A', 'B'), 'gene1^gene2'], (2 + 7) / 2)
    assert np.isclose(scores.loc[('B', 'A'), 'gene1^gene2'], (6 + 3) / 2)
    assert np.isclose(scores.loc[('A', 'A'), 'gene1^gene2'], (2 + 3) / 2)
    assert np.isclose(scores.loc[('B', 'B'), 'gene1^gene2'], (6 + 7) / 2)


def test_compute_communication_scores_multiple_pairs():
    """Test with multiple ligand-receptor pairs"""
    adata = create_controlled_adata()
    var_pairs = [('gene1', 'gene2'), ('gene2', 'gene3')]
    scores = compute_communication_scores(adata, 'group', var_pairs)

    assert list(scores.columns) == ['gene1^gene2', 'gene2^gene3']
    assert set(map(tuple, scores.index)) == {('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')}


def test_compute_communication_scores_errors():
    """Test error handling"""
    adata = create_controlled_adata()

    # Test non-existent gene
    with pytest.raises(ValueError, match='Variables not found'):
        compute_communication_scores(adata, 'group', [('nonexistent_gene', 'gene2')])

    # Test invalid scoring method
    with pytest.raises(KeyError):
        compute_communication_scores(adata, 'group', [('gene1', 'gene2')],
                                  communication_score='invalid_method')

    # Test invalid grouping column
    with pytest.raises(KeyError):
        compute_communication_scores(adata, 'nonexistent_group', [('gene1', 'gene2')])