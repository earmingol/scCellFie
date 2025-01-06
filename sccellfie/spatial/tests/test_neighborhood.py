import pytest
import numpy as np
from sccellfie.spatial.neighborhood import compute_neighbor_distribution
from sccellfie.datasets.toy_inputs import create_controlled_adata_with_spatial


@pytest.fixture
def spatial_adata():
    return create_controlled_adata_with_spatial()


def test_compute_neighbor_distribution_basic(spatial_adata):
    results = compute_neighbor_distribution(spatial_adata)

    assert isinstance(results, dict)
    assert all(k in results for k in ['radii', 'neighbors', 'quantiles', 'mean'])
    assert results['neighbors'].shape == (len(spatial_adata), 50)  # default n_points=50


def test_compute_neighbor_distribution_custom_params(spatial_adata):
    radius_range = (1.0, 5.0)
    n_points = 10
    quantiles = [0.1, 0.9]

    results = compute_neighbor_distribution(
        spatial_adata,
        radius_range=radius_range,
        n_points=n_points,
        quantiles=quantiles
    )

    assert len(results['radii']) == n_points
    assert results['radii'][0] >= radius_range[0]
    assert results['radii'][-1] <= radius_range[1]
    assert list(results['quantiles'].keys()) == quantiles


def test_compute_neighbor_distribution_values(spatial_adata):
    results = compute_neighbor_distribution(spatial_adata)

    # Test neighbor count matrix properties
    assert np.all(results['neighbors'] >= 0)  # No negative neighbors
    assert np.all(results['neighbors'][:, 0] <= len(spatial_adata) - 1)  # Max neighbors = n-1

    # Test quantile values
    assert np.all(results['quantiles'][0.05] <= results['quantiles'][0.95])


def test_compute_neighbor_distribution_custom_spatial_key(spatial_adata):
    spatial_adata.obsm['custom_spatial'] = spatial_adata.obsm['X_spatial']
    results = compute_neighbor_distribution(spatial_adata, spatial_key='custom_spatial')
    assert isinstance(results, dict)


def test_compute_neighbor_distribution_invalid_key(spatial_adata):
    with pytest.raises(KeyError):
        compute_neighbor_distribution(spatial_adata, spatial_key='invalid_key')