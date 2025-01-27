import numpy as np
from scipy.spatial import distance


def compute_neighbor_distribution(adata, radius_range=None, n_points=50, spatial_key='X_spatial', quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    Computes the distribution of neighbors per spot across different radii.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial coordinates.

    radius_range : tuple, optional (default: None)
        (min_radius, max_radius) to analyze. If None, automatically determined.

    n_points : int, optional (default: 50)
        Number of radius points to evaluate.

    spatial_key : str, optional (default: 'X_spatial')
        Key in adata.obsm containing spatial coordinates.

    quantiles : list, optional (default: [0.05, 0.25, 0.5, 0.75, 0.95])
        Quantiles to compute for the distribution

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'radii': array of evaluated radii
        - 'neighbors': matrix of neighbor counts (spots × radii)
        - 'quantiles': quantile values for each radius
        - 'mean': mean neighbors for each radius
    """
    # Extract spatial coordinates
    coords = adata.obsm[spatial_key]

    # Compute pairwise distances between spots
    dist_matrix = distance.pdist(coords)
    dist_matrix = distance.squareform(dist_matrix)

    # Determine radius range if not provided
    if radius_range is None:
        min_dist = np.floor(np.min(dist_matrix[dist_matrix > 0]))
        max_dist = np.ceil(np.percentile(dist_matrix, 95))  # 95th percentile to avoid outliers
        radius_range = (min_dist, max_dist)

    # Generate radius points
    radii = np.linspace(radius_range[0], radius_range[1], n_points)

    # Initialize matrix to store neighbor counts
    neighbor_counts = np.zeros((len(adata), len(radii)))

    # Compute number of neighbors for each spot at each radius
    for i, radius in enumerate(radii):
        neighbor_counts[:, i] = np.sum(dist_matrix <= radius, axis=1) - 1  # subtract self

    # Compute quantiles and mean
    quantile_values = np.percentile(neighbor_counts, [q * 100 for q in quantiles], axis=0)
    mean_neighbors = np.mean(neighbor_counts, axis=0)

    results = {'radii': radii,
               'neighbors': neighbor_counts,
               'quantiles': {q: qv for q, qv in zip(quantiles, quantile_values)},
               'mean': mean_neighbors
              }
    return results