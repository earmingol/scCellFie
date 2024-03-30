import pytest
import networkx as nx
import pandas as pd

from sccellfie.spatial.assortativity import compute_var_assortativity, compute_assortativity
from sccellfie.spatial.knn_network import create_knn_network
from sccellfie.tests.toy_inputs import create_random_adata_with_spatial


@pytest.mark.parametrize("use_raw", [False, True])
def test_compute_var_assortativity(use_raw):
    adata = create_random_adata_with_spatial(spatial_key='X_spatial')
    create_knn_network(adata, n_neighbors=2, spatial_key='X_spatial')
    assortativity = compute_var_assortativity(adata, var_name='gene1', use_raw=use_raw)
    assert isinstance(assortativity, float), "Output is not a float"


def test_compute_var_assortativity_pandas_graph():
    adata = create_random_adata_with_spatial(spatial_key='X_spatial')
    create_knn_network(adata, n_neighbors=2, spatial_key='X_spatial')
    adata.uns['spatial_network']['graph'] = nx.to_pandas_adjacency(adata.uns['spatial_network']['graph'])
    assortativity = compute_var_assortativity(adata, var_name='gene1')
    assert isinstance(assortativity, float), "Output is not a float"


def test_compute_var_assortativity_fail():
    adata = create_random_adata_with_spatial(spatial_key='X_spatial')
    with pytest.raises(ValueError):
        compute_var_assortativity(adata, var_name='gene1', use_raw=False)


def test_compute_assortativity():
    adata = create_random_adata_with_spatial(spatial_key='X_spatial')
    create_knn_network(adata, n_neighbors=2, spatial_key='X_spatial')
    compute_assortativity(adata, spatial_network_key='spatial_network', use_raw=False)
    assert isinstance(adata.uns['spatial_network']['assortativity'], pd.DataFrame), "Output is not a pandas DataFrame"
    assert all([v in adata.var_names for v in adata.uns['spatial_network']['assortativity']['Var-Name'].tolist()]), "Not all Var-Names are in var_names"