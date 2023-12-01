import numpy as np
import pandas as pd

import scanpy as sc


def create_test_adata(n_obs=100, n_vars=50, layers=None):
    # Create a simple AnnData object for testing
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(index=[f'cell{i}' for i in range(n_obs)])
    var = pd.DataFrame(index=[f'gene{i}' for i in range(n_vars)])
    adata = sc.AnnData(X=X, obs=obs, var=var)

    if layers:
        for layer in layers:
            adata.layers[layer] = np.random.rand(n_obs, n_vars)
    return adata