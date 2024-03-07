import os
import warnings

import scanpy as sc
import networkx as nx
import pandas as pd


def load_adata(folder, filename, reactions_filename=None, metabolic_tasks_filename=None, spatial_network_key='spatial_network', verbose=True):
    '''
    Loads an AnnData object and its scCellFie attributes from a folder.

    Parameters
    ----------
    folder: str
        The folder to load the AnnData object.

    filename: str
        The name of the file to load the AnnData object.

    reactions_filename: str, optional (default: None)
        The name of the file (without extension) to load the reactions object.
        If None, the default name is filename_reactions.

    metabolic_tasks_filename: str, optional (default: None)
        The name of the file  (without extension) to load the metabolic_tasks object.
        If None, the default name is filename_metabolic_tasks.

    spatial_network_key: str, optional (default: 'spatial_network')
        The key in adata.uns or a scCellFie_attribute.uns where the spatial
        knn graph is stored if exists.

    verbose: bool, optional (default: True)
        Whether to print the file names that were loaded.

    Returns
    -------
    adata: AnnData object
        Annotated data matrix.
        If scCellFie attributes are found, they are also loaded
        into adata.reactions and adata.metabolic_tasks.
    '''
    if reactions_filename is not None:
        rxn_filename = f'{folder}/{reactions_filename}.h5ad'
    else:
        rxn_filename = f'{folder}/{filename}_reactions.h5ad'

    if metabolic_tasks_filename is not None:
        mt_filename = f'{folder}/{metabolic_tasks_filename}.h5ad'
    else:
        mt_filename = f'{folder}/{filename}_metabolic_tasks.h5ad'

    # Load AnnData objects
    adata = sc.read_h5ad(f'{folder}/{filename}.h5ad')
    if verbose: print(f'{folder}/{filename}.h5ad was correctly loaded')
    if os.path.exists(rxn_filename):
        reactions = sc.read_h5ad(rxn_filename)
        if verbose: print(f'{rxn_filename} was correctly loaded')
    else:
        warnings.warn(f'{rxn_filename} not found. Skipping loading reactions.')
        reactions = None

    if os.path.exists(mt_filename):
        metabolic_tasks = sc.read_h5ad(mt_filename)
        if verbose: print(f'{mt_filename} was correctly loaded')
    else:
        warnings.warn(f'{mt_filename} not found. Skipping loading metabolic_tasks.')
        metabolic_tasks = None

    # Merge into adata
    if reactions is not None:
        adata.reactions = reactions
    if metabolic_tasks is not None:
        adata.metabolic_tasks = metabolic_tasks

    # Restore KNN graphs
    if spatial_network_key in adata.uns.keys():
        if isinstance(adata.uns[spatial_network_key]['graph'], pd.DataFrame):
            adata.uns[spatial_network_key]['graph'] = nx.from_pandas_adjacency(adata.uns[spatial_network_key]['graph'])
            if verbose: print(f"The graph in adata.uns['{spatial_network_key}']['graph'] was correctly loaded as a networkx.Graph object")
    if hasattr(adata, 'reactions'):
        if spatial_network_key in adata.reactions.uns.keys():
            if isinstance(adata.reactions.uns[spatial_network_key]['graph'], pd.DataFrame):
                adata.reactions.uns[spatial_network_key]['graph'] = nx.from_pandas_adjacency(adata.reactions.uns[spatial_network_key]['graph'])
                if verbose: print(f"The graph in adata.reactions.uns['{spatial_network_key}']['graph'] was correctly loaded as a networkx.Graph object")
    if hasattr(adata, 'metabolic_tasks'):
        if spatial_network_key in adata.metabolic_tasks.uns.keys():
            if isinstance(adata.metabolic_tasks.uns[spatial_network_key]['graph'], pd.DataFrame):
                adata.metabolic_tasks.uns[spatial_network_key]['graph'] = nx.from_pandas_adjacency(adata.metabolic_tasks.uns[spatial_network_key]['graph'])
                if verbose: print(f"The graph in adata.metabolic_tasks.uns['{spatial_network_key}']['graph'] was correctly loaded as a networkx.Graph object")
    return adata