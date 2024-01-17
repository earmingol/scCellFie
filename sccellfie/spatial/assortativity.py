import networkx as nx
import numpy as np
import pandas as pd

from tqdm import tqdm


def compute_var_assortativity(adata, var_name, use_raw=False):
    '''
    Computes the assortativity of a variable (e.g., a gene, a metabolic tasks, etc.) in a spatial network.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    var_name: str
        The name of the variable to compute assortativity.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    assortativity: float
        The assortativity of the variable in the spatial network.
    '''
    if 'spatial_network' not in adata.uns.keys():
        raise ValueError('spatial_network not found in adata.uns. Run sccellfie.spatial.knn_network.create_knn_network() first.')
    H = adata.uns['spatial_network']['graph'].copy()

    if use_raw:
        X = adata.raw.X
    else:
        X = adata.X

    _idx = np.argwhere(adata.var_names == var_name).item()

    # Add weights to nodes. This is the value in adata.X or adata.raw.X for each node
    weights = dict(zip(adata.obs_names, X[:, _idx].flatten()))
    nx.set_node_attributes(H, weights, 'weight')

    # Compute assortativity
    assortativity = nx.numeric_assortativity_coefficient(H, 'weight')
    return assortativity


def compute_assortativity(adata, use_raw=False):
    '''
    Computes the assortativity of all variables (e.g., genes, metabolic tasks, etc.) in a spatial network.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    None
        A pandas.DataFrame object containing the assortativity of each variable in the spatial network
        is added to adata.uns['spatial_network']['assortativity'].
    '''
    records = []
    for var_ in tqdm(adata.var_names):
        assort = compute_var_assortativity(adata, var_name=var_, use_raw=use_raw)
        records.append((var_, assort))

    assortativity_df = pd.DataFrame.from_records(records, columns=['Var-Name', 'Assortativity'])
    adata.uns['spatial_network']['assortativity'] = assortativity_df