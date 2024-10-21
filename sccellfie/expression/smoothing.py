import numpy as np
import scipy.sparse as sp

from tqdm import tqdm


def get_smoothing_matrix(adata, mode, neighbors_key='neighbors'):
    """
    Calculates the smoothing matrix S based on the nearest neighbor graph in adata.obsp.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix containing the nearest neighbor graph.

    mode : str
        The mode for calculating the smoothing matrix. Can be either 'adjacency' or 'connectivity'.

    neighbors_key : str, optional (default: 'neighbors')
        The key in adata.uns where the information about the pre-run KNN analysis was stored.
        This key points to a dictionary containing the 'connectivities_key', 'distances_key',
        and 'params' from the analysis.

    Returns
   -------
    S : numpy.ndarray or scipy.sparse.csr_matrix
        The smoothing matrix S such that S @ X smoothes the signal X over neighbors.
        If the input matrix A is sparse, S will be returned as a scipy.sparse.csr_matrix.
        Otherwise, S will be a numpy.ndarray.

    Raises
    ------
    ValueError
        If an unknown mode is provided.
    """
    if mode == 'adjacency':
        distances_key = adata.uns[neighbors_key]['distances_key']
        A = (adata.obsp[distances_key] > 0).astype(int)
    elif mode == 'connectivity':
        connectivities_key = adata.uns[neighbors_key]['connectivities_key']
        A = adata.obsp[connectivities_key]
    else:
        raise ValueError(f'unknown mode {mode}')

    update = False
    if sp.issparse(A):
        A = A.toarray()
        update = True
    # Normalize the smoothing matrix
    norm_vec = A.sum(axis=1)
    norm_vec[norm_vec == 0] = 1  # Avoid division by zero
    S = A / norm_vec[:, np.newaxis]
    if update:
        S = sp.csr_matrix(S)
    return S


def smooth_expression_knn(adata, key_added='smoothed_X', neighbors_key='neighbors', mode='connectivity', alpha=0.33,
                          n_chunks=None, chunk_size=None, use_raw=False, disable_pbar=False):
    """
    Smooths expression values based on KNNs of single cells using Scanpy.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix containing the expression data and nearest neighbor graph.

    key_added : str, optional (default: 'smoothed_X')
        The key in adata.layers where the smoothed expression matrix will be stored.

    neighbors_key : str, optional (default: 'neighbors')
        The key in adata.uns where the information about the pre-run KNN analysis was stored.
        This key points to a dictionary containing the 'connectivities_key', 'distances_key',
        and 'params' from the analysis.

    mode : str, optional (default: 'connectivity')
        The mode for calculating the smoothing matrix. Can be either 'adjacency' or 'connectivity'.

    alpha : float, optional (default: 0.33)
        The weight or fraction of the smoothed expression to use in the final expression matrix.
        The final expression matrix is computed as (1 - alpha) * X + alpha * (S @ X), where X is the
        original expression matrix and S is the smoothed matrix.

    n_chunks : int, optional (default: None)
        The number of chunks to split the cells into for processing. If not provided, chunk_size is used.

    chunk_size : int, optional (default: None)
        The size of each chunk of cells to process. If not provided, n_chunks is used.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    disable_pbar: bool, optional (default: False)
        Whether to disable the progress bar.

    Returns
    -------
    None
        The smoothed expression matrix is stored in adata.layers[key_added].

    Notes
    -----
    This function smoothes the expression values of single cells based on their K-nearest neighbors (KNNs)
    using the Scanpy package. The smoothing is performed by calculating a smoothing matrix S based on the
    nearest neighbor graph and then computing the smoothed expression as (1 - alpha) * X + alpha * (S @ X),
    where X is the original expression matrix.

    The smoothing is performed in chunks to reduce memory usage. The number of chunks or the chunk size
    can be specified using the n_chunks or chunk_size parameters, respectively.

    The smoothed expression matrix is stored in adata.layers[key_added].
    """
    # Get the connectivities matrix
    connectivities = adata.obsp[adata.uns[neighbors_key]['connectivities_key']]

    # Determine the number of chunks and chunk size
    n_cells = adata.n_obs
    if n_chunks is None and chunk_size is None:
        n_chunks = 1  # Default number of chunks
    if n_chunks is not None:
        chunk_size = int(np.ceil(n_cells / n_chunks))
    else:
        n_chunks = int(np.ceil(n_cells / chunk_size))

    # Initialize the smoothed expression matrix
    if use_raw:
        X = adata.raw.X
    else:
        X = adata.X

    if isinstance(X, sp.coo_matrix):
        X = X.tocsr()

    smoothed_matrix = np.zeros(X.shape)

    # Iterate over chunks of cells
    for i in tqdm(range(n_chunks), disable=disable_pbar, desc='Smoothing Expression'):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_cells)

        # Get the connectivities for the current chunk
        chunk_connectivities = connectivities[start_idx:end_idx, :]

        # Find the unique neighbor indices for the current chunk
        neighbor_indices = np.unique(chunk_connectivities.nonzero()[1])

        # Create a set of cell indices including the chunk and its neighbors
        chunk_and_neighbors = np.union1d(np.arange(start_idx, end_idx), neighbor_indices)

        # Subset the adata based on the chunk and neighbor indices
        subset_adata = adata[chunk_and_neighbors, :]

        # Get the smoothing matrix for the current chunk and its neighbors
        smoothing_mat = get_smoothing_matrix(subset_adata, mode, neighbors_key=neighbors_key)

        # Get the expression data for the current chunk and its neighbors
        chunk_expression = X[chunk_and_neighbors, :]

        # Compute the expression data purely based on cell neighbors
        chunk_smoothed = smoothing_mat @ chunk_expression
        if sp.issparse(X):
            chunk_smoothed = chunk_smoothed.toarray()
            X_ = X.toarray()
        else:
            X_ = X

        # Extract the smoothed expression for the cells in the current chunk
        chunk_indices = np.arange(start_idx, end_idx)
        subset_mapping = dict(zip(chunk_and_neighbors, range(len(chunk_and_neighbors))))
        smoothed_chunk_indices = [subset_mapping[i] for i in chunk_indices]

        # Smooth by alpha
        smoothed_matrix[chunk_indices, :] = (1. - alpha) * X_[chunk_indices, :] + alpha * chunk_smoothed[smoothed_chunk_indices, :]

    # Store the smoothed expression matrix in adata.layers
    if sp.issparse(adata.X):
        smoothed_matrix = sp.csr_matrix(smoothed_matrix)
    adata.layers[key_added] = smoothed_matrix