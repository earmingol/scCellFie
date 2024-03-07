import networkx as nx
from sccellfie.spatial.knn_network import find_spatial_neighbors, create_knn_network
from sccellfie.tests.toy_inputs import create_controlled_adata_with_spatial

def test_find_spatial_neighbors():
    # Create mock data with spatial coordinates
    adata = create_controlled_adata_with_spatial()

    # Expected number of neighbors
    n_neighbors = 1

    # Run the function
    find_spatial_neighbors(adata, n_neighbors=n_neighbors)

    # Check if the spatial neighbors keys were added correctly
    assert 'spatial_neighbors' in adata.uns
    assert 'spatial_connectivities' in adata.obsp
    assert 'spatial_distances' in adata.obsp

    # Check if the parameters are set correctly
    params = adata.uns['spatial_neighbors']['params']
    assert params['n_neighbors'] == n_neighbors

    # Check if the spatial graph has the correct number of edges
    # Each cell should have at most n_neighbors connections
    connectivities = adata.obsp['spatial_connectivities']
    assert connectivities.getnnz(axis=1).max() == n_neighbors, "Unexpected number of neighbors"


def test_create_knn_network():
    # Create mock data with spatial coordinates
    adata = create_controlled_adata_with_spatial()

    # Expected number of neighbors
    n_neighbors = 1

    # Run the function
    create_knn_network(adata, n_neighbors=n_neighbors)

    # Check if the spatial network key was added correctly
    added_key = 'spatial_network'
    assert added_key in adata.uns

    # Get the network graph
    G = adata.uns[added_key]['graph']

    # Check if the parameters in the graph are set correctly
    assert adata.uns[added_key]['params']['n_neighbors'] == n_neighbors, "n_neighbors in graph parameters is not equal to n_neighbors"

    # Check if the graph has the correct number of nodes and edges
    assert len(G.nodes()) == adata.n_obs

    # Each node should have at most n_neighbors edges, accounting for undirected edges
    assert all(len(list(G.neighbors(node))) == n_neighbors for node in G.nodes()), "Number of neighbors in graph is not equal to n_neighbors"

def test_different_neighbors_in_params():
    # Create mock data with spatial coordinates
    adata = create_controlled_adata_with_spatial()

    # Expected number of neighbors
    n_neighbors = 1

    # Run the function
    find_spatial_neighbors(adata, n_neighbors=n_neighbors)
    adata.uns['spatial_neighbors']['params']['n_neighbors'] = 5
    create_knn_network(adata, n_neighbors=n_neighbors)

    # Check if the spatial network key was added correctly
    added_key = 'spatial_network'
    assert added_key in adata.uns

    # Get the network graph
    G = adata.uns[added_key]['graph']

    # Check if the parameters in the graph are set correctly
    assert adata.uns[added_key]['params']['n_neighbors'] == n_neighbors, "n_neighbors in graph parameters is not equal to n_neighbors"

    # Check if the graph has the correct number of nodes and edges
    assert len(G.nodes()) == adata.n_obs

    # Each node should have at most n_neighbors edges, accounting for undirected edges
    assert all(len(list(G.neighbors(node))) == n_neighbors for node in G.nodes()), "Number of neighbors in graph is not equal to n_neighbors"


