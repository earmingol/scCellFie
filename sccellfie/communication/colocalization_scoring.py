import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr

from sccellfie.expression.aggregation import AGG_FUNC


def compute_local_colocalization_scores(adata, var1, var2, neighbors_radius, method='pairwise_concordance', spatial_key='X_spatial',
                                        min_neighbors=3, threshold1=None, threshold2=None, score_key=None, inplace=True):
    """
    Computes local colocalization scores between two variables for each spatial spot.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data and spatial coordinates.

    var1 : str
        Name of first variable to analyze.

    var2 : str
        Name of second variable to analyze.

    neighbors_radius : float
        Radius for assigning a neighborhood of a spot
        (neighbors within this radius are considered, and the sport is the center).

    method : str, optional (default: 'pairwise_concordance')
        Method to compute colocalization:
        - 'correlation': Local Pearson correlation between var1 and var2 across spot & neighbors.
        - 'concordance': Compute the fraction of spots where both genes are expressed above their thresholds.
        - 'pairwise_concordance': Compute the fraction of spot pairs in the neighborhood where var1 and var2 are expressed above their thresholds in sport 1 and 2, respectively.
        - 'cosine': Local cosine similarity between var1 and var2 across spot & neighbors.
        - 'weighted_gmean': Local weighted geometric mean across spot & neighbors (weighted by distance).
        - 'regularized_weighted_gmean': Local regularized and weighted geometric mean across spot & neighbors (weighted by distance).

    spatial_key : str, optional (default: 'spatial')
        Key in adata.obsm containing spatial coordinates

    min_neighbors : int, optional (default: 3)
        Minimum number of neighbors required for computing score. If less neighbors are found, score is NaN.

    threshold1 : float, optional (default: None)
        Threshold for var1. If None, the mean of var1 is used.

    threshold2 : float, optional (default: None)
        Threshold for var2. If None, the mean of var2 is used.

    score_key : str, optional (default: None)
        Key to store the computed colocalization scores in adata.obs. If None, a default key is used.

    inplace : bool, optional (default: True)
        If True, the computed scores are added to adata.obs. Otherwise, the scores are returned as a numpy array.

    Returns
    -------
    numpy.ndarray
        Array of colocalization scores for each spot
    """
    # Extract spatial coordinates and expression values
    coords = adata.obsm[spatial_key]
    expr1 = adata[:, var1].X.toarray().flatten()
    expr2 = adata[:, var2].X.toarray().flatten()

    # Normalize expression values to [0,1] range
    expr1_norm = (expr1 - np.min(expr1)) / (np.max(expr1) - np.min(expr1))
    expr2_norm = (expr2 - np.min(expr2)) / (np.max(expr2) - np.min(expr2))

    # Compute pairwise distances between spots
    dist_matrix = distance.pdist(coords)
    dist_matrix = distance.squareform(dist_matrix)

    # Initialize scores array
    scores = np.zeros(len(adata)) * np.nan

    if method == 'correlation':
        # Local Pearson correlation within radius for each spot
        for i in range(len(adata)):
            neighbors = np.where(dist_matrix[i] <= neighbors_radius)[0]
            if len(neighbors) >= min_neighbors:
                score, _ = pearsonr(expr1[neighbors], expr2[neighbors])
                scores[i] = score

    elif method == 'concordance':
        if threshold1 is None:
            threshold1 = AGG_FUNC['mean'](expr1, axis=None)
        if threshold2 is None:
            threshold2 = AGG_FUNC['mean'](expr2, axis=None)
        # Compute local concordance of expression patterns
        for i in range(len(adata)):
            neighbors = np.where(dist_matrix[i] <= neighbors_radius)[0] # This should already include itself
            if len(neighbors) >= min_neighbors:
                # Compare if both genes are similarly high in neighborhood
                concordant = np.sum(
                    ((expr1[neighbors] > threshold1) & (expr2[neighbors] > threshold2))
                )
                scores[i] = concordant / len(neighbors)

    elif method == 'pairwise_concordance':
        if threshold1 is None:
            threshold1 = AGG_FUNC['mean'](expr1, axis=None)
        if threshold2 is None:
            threshold2 = AGG_FUNC['mean'](expr2, axis=None)

        # Compute pairwise concordance for each spot's neighborhood
        for i in range(len(adata)):
            neighbors = np.where(dist_matrix[i] <= neighbors_radius)[0]
            if len(neighbors) >= min_neighbors:
                concordant_pairs = 0
                total_pairs = 0

                # Compare each pair of spots in the neighborhood
                for idx1 in neighbors:
                    for idx2 in neighbors:
                        # Check if var1 in first spot and var2 in second spot exceed thresholds
                        if expr1[idx1] > threshold1 and expr2[idx2] > threshold2:
                            concordant_pairs += 1
                        total_pairs += 1

                scores[i] = concordant_pairs / total_pairs

    elif method == 'cosine':
        # Compute product of normalized intensities in neighborhood
        for i in range(len(adata)):
            neighbors = np.where(dist_matrix[i] <= neighbors_radius)[0]
            if len(neighbors) >= min_neighbors:
                # Weight by distance
                weights = 1 / (1 + dist_matrix[i][neighbors])
                weights = weights / np.sum(weights)

                # Compute weighted product of normalized intensities
                local_score = np.sum(
                    weights * expr1_norm[neighbors] * expr2_norm[neighbors]
                ) / (np.sqrt(np.sum(weights * expr1_norm[neighbors] * expr1_norm[neighbors])) * np.sqrt(
                    np.sum(weights * expr2_norm[neighbors] * expr2_norm[neighbors])))
                scores[i] = local_score

    elif method in ['weighted_gmean', 'regularized_weighted_gmean']:
        # Compute product of normalized intensities in neighborhood
        for i in range(len(adata)):
            neighbors = np.where(dist_matrix[i] <= neighbors_radius)[0]
            if len(neighbors) >= min_neighbors:
                # Weight by distance
                weights = 1 / (1 + dist_matrix[i][neighbors])
                weights = weights / np.sum(weights)

                # Compute weighted product of normalized intensities
                local_score = np.sum(
                    weights * np.sqrt(expr1_norm[neighbors] * expr2_norm[neighbors])
                )
                scores[i] = local_score
        if method == 'regularized_weighted_gmean':
            scores = scores / (scores + AGG_FUNC['mean'](scores, axis=None))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Add scores to adata
    if inplace:
        if score_key is None:
            score_key = f'{var1}^{var2}'
        adata.obs[score_key] = scores
    else:
        return scores