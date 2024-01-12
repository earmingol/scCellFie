import numpy as np
import pandas as pd

import scanpy as sc
from scipy import sparse


def create_random_adata(n_obs=100, n_vars=50, n_clusters=5, layers=None):
    # Create a simple AnnData object for testing
    X = np.random.randint(low=0, high=100, size=(n_obs, n_vars))
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
    adata.obs['group'] = ['A', 'A', 'B', 'B']
    adata.X = sparse.csr_matrix(adata.X)
    adata.raw = adata.copy()
    return adata


def create_controlled_gpr_dict():
    gpr_dict = {'rxn1': 'gene1',
                'rxn2': 'gene2',
                'rxn3': 'gene2 and gene3',
                'rxn4': 'gene1 or gene3'
                }
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


def create_global_threshold(threshold=0.5, n_vars=4):
    thresholds = pd.DataFrame(index=[f'gene{i}' for i in range(1, n_vars+1)])
    thresholds['global'] = [threshold]*n_vars
    return thresholds
