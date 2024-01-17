import warnings

import pandas as pd
import scanpy as sc
import squidpy as sq
import networkx as nx



def find_spatial_neighbors(adata, n_neighbors=10, metric='euclidean', spatial_key='X_spatial'):
    '''
    Finds the spatial neighbors of each cell in adata.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    n_neighbors: int, optional (default: 10)
        The number of neighbors to use for k-nearest neighbors.

    metric: str, optional (default: 'euclidean')
        The distance metric to use for k-nearest neighbors.

    spatial_key: str, optional (default: 'X_spatial')
        The key in adata.obsm where the spatial coordinates are stored.

    Returns
    -------
    None
        The spatial neighbors are added to adata.uns['spatial_neighbors'].
    '''
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=spatial_key, metric=metric, key_added='spatial')
    adata.uns['spatial_neighbors'] = {'connectivities_key': 'spatial_connectivities',
                                      'distances_key': 'spatial_distances',
                                      'params': {'n_neighbors': n_neighbors, 'metric': metric}
                                      }


def create_knn_network(adata, n_neighbors=10, spatial_key='X_spatial'):
    '''
    Creates a k-nearest neighbor network from the spatial coordinates in adata.obsm[obsm_key].

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    n_neighbors: int, optional (default: 10)
        The number of neighbors to use for k-nearest neighbors.

    spatial_key: str, optional (default: 'X_spatial')
        The key in adata.obsm where the spatial coordinates are stored.

    Returns
    -------
    None
        The k-nearest neighbor network is added to adata.uns['spatial_network'].
    '''
    if 'spatial_connectivities' not in adata.obsp.keys():
        warnings.warn('spatial_connectivities not found in adata.uns. Creating spatial_neighbors.')
        sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, n_neighs=n_neighbors, key_added='spatial')
    elif adata.uns['spatial_neighbors']['params']['n_neighbors'] != n_neighbors:
        warnings.warn("n_neighbors in adata.uns['spatial_neighbors']['params'] is different from n_neighbors. "
                      "Creating spatial_connectivities.")
        sq.gr.spatial_neighbors(adata, spatial_key=spatial_key, n_neighs=n_neighbors, key_added='spatial')

    A = pd.DataFrame.sparse.from_spmatrix(adata.obsp['spatial_connectivities'], index=adata.obs_names, columns=adata.obs_names)

    # Create network
    G = nx.from_pandas_adjacency(A)
    adata.uns['spatial_network'] = {'graph': G, 'params': {'n_neighbors': n_neighbors}}