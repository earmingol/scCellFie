import cobra
import numpy as np
import pandas as pd

import scanpy as sc
from scipy import sparse


def create_random_adata(n_obs=100, n_vars=50, n_clusters=5, layers=None):
    # Create a simple AnnData object for testing
    X = np.random.randint(low=0, high=100, size=(n_obs, n_vars)).astype(float)
    obs = pd.DataFrame(index=[f'cell{i}' for i in range(1, n_obs+1)])
    var = pd.DataFrame(index=[f'gene{i}' for i in range(1, n_vars+1)])
    obs['cluster'] = pd.Categorical([f'cluster{i}' for i in np.random.randint(1, n_clusters+1, size=n_obs)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.X = sparse.csr_matrix(adata.X)
    adata.raw = adata.copy()

    if layers:
        if isinstance(layers, str):
            layers = [layers]
        for layer in layers:
            adata.layers[layer] = np.random.rand(n_obs, n_vars)
    return adata


def create_random_adata_with_spatial(n_obs=100, n_vars=50, n_clusters=5, layers=None, spatial_key='X_spatial', n_cols=5):
    # Create a simple AnnData object for testing
    X = np.random.randint(low=0, high=100, size=(n_obs, n_vars)).astype(float)
    obs = pd.DataFrame(index=[f'cell{i}' for i in range(1, n_obs + 1)])
    var = pd.DataFrame(index=[f'gene{i}' for i in range(1, n_vars + 1)])
    obs['cluster'] = pd.Categorical([f'cluster{i}' for i in np.random.randint(1, n_clusters + 1, size=n_obs)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.X = sparse.csr_matrix(adata.X)
    grid = [[row, col] for row in range(1, int(np.ceil(n_obs / n_cols)) + 1) for col in range(1, n_cols + 1)]
    adata.obsm[spatial_key] = np.array(grid[:n_obs]).astype(float)
    adata.raw = adata.copy()

    if layers:
        if isinstance(layers, str):
            layers = [layers]
        for layer in layers:
            adata.layers[layer] = np.random.rand(n_obs, n_vars)
    return adata


def create_controlled_adata():
    # Create a small, controlled AnnData object
    data = np.array([
        [1, 2, 0],  # Cell1
        [3, 4, 2],  # Cell2
        [5, 6, 10],  # Cell3
        [7, 8, 6],  # Cell4
    ])
    adata = sc.AnnData(X=data)
    adata.var_names = ['gene1', 'gene2', 'gene3']
    adata.obs_names = ['cell1', 'cell2', 'cell3', 'cell4']
    adata.obs['group'] = pd.Categorical(['A', 'A', 'B', 'B'])
    adata.X = sparse.csr_matrix(adata.X)
    adata.raw = adata.copy()
    return adata


def create_controlled_adata_with_spatial():
    # Create a small, controlled AnnData object
    data = np.array([
        [1, 2, 0],  # Cell1
        [3, 4, 2],  # Cell2
        [5, 6, 10],  # Cell3
        [7, 8, 6],  # Cell4
    ])
    adata = sc.AnnData(X=data)
    adata.var_names = ['gene1', 'gene2', 'gene3']
    adata.obs_names = ['cell1', 'cell2', 'cell3', 'cell4']
    adata.obs['group'] = pd.Categorical(['A', 'A', 'B', 'B'])
    adata.X = sparse.csr_matrix(adata.X)
    adata.obsm['X_spatial'] = np.array([[0, 0], [1, 1], [3, 3], [4, 4]]).astype(float)
    adata.raw = adata.copy()
    return adata


def add_toy_neighbors(adata, n_neighbors=10):
    """
    Add a toy neighbor object to the AnnData object, mimicking scanpy's format.
    """
    adata_ = adata.copy()
    n_obs = adata_.n_obs

    # Create toy connectivities and distances matrices
    connectivities = sparse.csr_matrix((np.ones(n_obs * n_neighbors),
                                       (np.repeat(np.arange(n_obs), n_neighbors),
                                       np.random.choice(n_obs, n_obs * n_neighbors))),
                                       shape=(n_obs, n_obs))

    distances = sparse.csr_matrix((np.random.rand(n_obs * n_neighbors),
                                  (np.repeat(np.arange(n_obs), n_neighbors),
                                  np.random.choice(n_obs, n_obs * n_neighbors))),
                                  shape=(n_obs, n_obs))

    # Create the neighbors dictionary
    adata_.uns['neighbors'] = {
        'params': {
            'n_neighbors': n_neighbors,
            'method': 'umap'
        },
        'connectivities_key': 'connectivities',
        'distances_key': 'distances'
    }

    # Add matrices to obsp
    adata_.obsp['connectivities'] = connectivities
    adata_.obsp['distances'] = distances

    # Add toy PCA and UMAP
    adata_.obsm['X_pca'] = np.random.rand(n_obs, 50)  # 50 PCA components
    adata_.obsm['X_umap'] = np.random.rand(n_obs, 2)  # 2D UMAP
    return adata_


def create_controlled_gpr_dict():
    gpr_dict = {'rxn1': 'gene1',
                'rxn2': 'gene2',
                'rxn3': 'gene2 and gene3',
                'rxn4': 'gene1 or gene3'
                }

    # Initialize GPRs
    gpr_dict = {k: cobra.core.gene.GPR().from_string(gpr) for k, gpr in gpr_dict.items()}
    return gpr_dict


def create_controlled_task_by_rxn():
    task_by_rxn = pd.DataFrame(data=[[1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [1, 0, 0, 0],
                                     [1, 1, 1, 1],
                                     ],
                               index=['task1', 'task2', 'task3', 'task4'],
                               columns=['rxn1', 'rxn2', 'rxn3', 'rxn4'])
    return task_by_rxn


def create_controlled_rxn_by_gene():
    rxn_by_gene = pd.DataFrame(data=[[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 1, 1],
                                     [1, 0, 1],
                                     ],
                               index=['rxn1', 'rxn2', 'rxn3', 'rxn4'],
                               columns=['gene1', 'gene2', 'gene3'])
    return rxn_by_gene


def create_controlled_task_by_gene():
    task_by_gene = pd.DataFrame(data=[[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 0, 0],
                                      [1, 1, 1],
                                      ],
                                index=['task1', 'task2', 'task3', 'task4'],
                                columns=['gene1', 'gene2', 'gene3'])
    return task_by_gene


def create_global_threshold(threshold=0.5, n_vars=4):
    thresholds = pd.DataFrame(index=[f'gene{i}' for i in range(1, n_vars+1)])
    thresholds['global'] = [threshold]*n_vars
    return thresholds