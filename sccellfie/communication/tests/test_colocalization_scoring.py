import pytest
import numpy as np
from scipy import sparse

from sccellfie.communication.colocalization_scoring import compute_local_colocalization_scores
from sccellfie.datasets.toy_inputs import create_controlled_adata_with_spatial


def test_compute_local_colocalization_basic():
    adata = create_controlled_adata_with_spatial()
    # Use larger radius since spatial coords are [0,0], [1,1], [3,3], [4,4]
    scores = compute_local_colocalization_scores(
        adata, 'gene1', 'gene2', neighbors_radius=3.0,
        method='correlation', min_neighbors=2, inplace=False
    )

    assert isinstance(scores, np.ndarray)
    assert len(scores) == adata.n_obs
    assert not np.all(np.isnan(scores))


def test_compute_local_colocalization_methods():
    adata = create_controlled_adata_with_spatial()
    methods = ['correlation', 'concordance', 'pairwise_concordance',
               'cosine', 'weighted_gmean', 'regularized_weighted_gmean']

    for method in methods:
        scores = compute_local_colocalization_scores(
            adata, 'gene1', 'gene2', neighbors_radius=3.0,
            method=method, min_neighbors=2, inplace=False
        )
        assert not np.all(np.isnan(scores)), f"Method {method} returned all NaN"
        assert len(scores) == adata.n_obs


def test_compute_local_colocalization_thresholds():
    adata = create_controlled_adata_with_spatial()
    scores1 = compute_local_colocalization_scores(
        adata, 'gene1', 'gene2', neighbors_radius=3.0,
        method='concordance', threshold1=2.0, threshold2=3.0,
        min_neighbors=2, inplace=False
    )

    scores2 = compute_local_colocalization_scores(
        adata, 'gene1', 'gene2', neighbors_radius=3.0,
        method='concordance', min_neighbors=2, inplace=False
    )

    assert not np.array_equal(scores1, scores2)


def test_compute_local_colocalization_inplace():
    adata = create_controlled_adata_with_spatial()
    compute_local_colocalization_scores(
        adata, 'gene1', 'gene2', neighbors_radius=3.0,
        method='correlation', score_key='test_score',
        min_neighbors=2
    )

    assert 'test_score' in adata.obs.columns


def test_compute_local_colocalization_min_neighbors():
    adata = create_controlled_adata_with_spatial()
    scores = compute_local_colocalization_scores(
        adata, 'gene1', 'gene2', neighbors_radius=0.5,
        min_neighbors=3, inplace=False
    )

    assert np.all(np.isnan(scores))


def test_compute_local_colocalization_errors():
    adata = create_controlled_adata_with_spatial()

    with pytest.raises(ValueError):
        compute_local_colocalization_scores(
            adata, 'gene1', 'gene2', neighbors_radius=3.0,
            method='invalid_method'
        )

    with pytest.raises(KeyError):
        compute_local_colocalization_scores(
            adata, 'nonexistent_gene', 'gene2', neighbors_radius=3.0
        )


def test_compute_local_colocalization_different_spatial_key():
    adata = create_controlled_adata_with_spatial()
    # Create custom coordinates with different relative distances
    custom_coords = np.array([[0, 0], [2, 2], [5, 5], [8, 8]]).astype(float)
    adata.obsm['custom_spatial'] = custom_coords

    scores1 = compute_local_colocalization_scores(
        adata, 'gene1', 'gene2', neighbors_radius=4.0,
        spatial_key='custom_spatial', min_neighbors=2, method='correlation', inplace=False
    )

    scores2 = compute_local_colocalization_scores(
        adata, 'gene1', 'gene2', neighbors_radius=2.0,
        min_neighbors=2, method='correlation', inplace=False
    )

    assert not np.array_equal(scores1, scores2)