import pytest
import numpy as np
import pandas as pd

from sccellfie.communication.traditional_scoring import compute_communication_scores
from sccellfie.datasets.toy_inputs import create_controlled_adata


def test_compute_communication_scores_basic():
    adata = create_controlled_adata()
    var_pairs = [('gene1', 'gene2')]
    scores = compute_communication_scores(adata, 'group', var_pairs)

    assert isinstance(scores, pd.DataFrame)
    assert list(scores.columns) == ['gene1^gene2']
    assert set(scores.index) == {'A', 'B'}
    assert not scores.isna().any().any()


def test_compute_communication_scores_values():
    adata = create_controlled_adata()
    var_pairs = [('gene1', 'gene2')]

    # Test gmean
    scores = compute_communication_scores(adata, 'group', var_pairs)
    assert np.isclose(scores.loc['A', 'gene1^gene2'], np.sqrt(2 * 3))
    assert np.isclose(scores.loc['B', 'gene1^gene2'], np.sqrt(6 * 7))

    # Test product
    scores = compute_communication_scores(adata, 'group', var_pairs, communication_score='product')
    assert np.isclose(scores.loc['A', 'gene1^gene2'], 2 * 3)
    assert np.isclose(scores.loc['B', 'gene1^gene2'], 6 * 7)

    # Test mean
    scores = compute_communication_scores(adata, 'group', var_pairs, communication_score='mean')
    assert np.isclose(scores.loc['A', 'gene1^gene2'], (2 + 3) / 2)
    assert np.isclose(scores.loc['B', 'gene1^gene2'], (6 + 7) / 2)


def test_compute_communication_scores_errors():
    adata = create_controlled_adata()

    with pytest.raises(ValueError):
        compute_communication_scores(adata, 'group', [('nonexistent_gene', 'gene2')])

    with pytest.raises(KeyError):
        compute_communication_scores(adata, 'group', [('gene1', 'gene2')],
                                     communication_score='invalid_method')