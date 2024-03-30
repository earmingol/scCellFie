import warnings
import networkx as nx
from pathlib import Path


def save_adata(adata, folder, filename, spatial_network_key='spatial_network', verbose=True):
    '''
    Saves an AnnData object and its scCellFie attributes to a folder.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    folder: str
        The folder to save the AnnData object.

    filename: str
        The name of the file to save the AnnData object. Do not include the file extension.

    spatial_network_key: str, optional (default: 'spatial_network')
        The key in adata.uns or a scCellFie_attribute.uns where the spatial knn graph is stored.

    verbose: bool, optional (default: True)
        Whether to print the file names that were saved.

    Returns
    -------
    None
        The AnnData object is saved to folder/filename.h5ad.
        The scCellFie attributes are saved to:
            - reactions: folder/filename_reactions.h5ad.
            - metabolic_tasks: folder/filename_metabolic_tasks.h5ad.
    '''
    # Check folder path
    Path(folder).mkdir(parents=True, exist_ok=True)
    if spatial_network_key in adata.uns.keys():
        if isinstance(adata.uns[spatial_network_key]['graph'], nx.Graph):
            adata.uns[spatial_network_key]['graph'] = nx.to_pandas_adjacency(adata.uns[spatial_network_key]['graph'])
            warn = f"adata.uns['{spatial_network_key}']['graph'] was converted from a networkx.Graph object to a pandas adjacency matrix to be saved with the AnnData object."
            warnings.warn(warn)

    adata_filename = f'{folder}/{filename}.h5ad'
    adata.write_h5ad(adata_filename)
    if verbose: print(f'{adata_filename} was correctly saved')

    if hasattr(adata, 'reactions'):
        if spatial_network_key in adata.reactions.uns.keys():
            if isinstance(adata.reactions.uns[spatial_network_key]['graph'], nx.Graph):
                adata.reactions.uns[spatial_network_key]['graph'] = nx.to_pandas_adjacency(adata.reactions.uns[spatial_network_key]['graph'])
                warn = f"adata.reactions.uns['{spatial_network_key}']['graph'] was converted from a networkx.Graph object to a pandas adjacency matrix to be saved with the AnnData object."
                warnings.warn(warn)
        reaction_filename = f'{folder}/{filename}_reactions.h5ad'
        adata.reactions.write_h5ad(reaction_filename)
        if verbose: print(f'{reaction_filename} was correctly saved')
    else:
        warnings.warn('No adata.reactions found. Skipping saving reactions.')
    if hasattr(adata, 'metabolic_tasks'):
        if spatial_network_key in adata.metabolic_tasks.uns.keys():
            if isinstance(adata.metabolic_tasks.uns[spatial_network_key]['graph'], nx.Graph):
                adata.metabolic_tasks.uns[spatial_network_key]['graph'] = nx.to_pandas_adjacency(adata.metabolic_tasks.uns[spatial_network_key]['graph'])
                warn = f"adata.metabolic_tasks.uns['{spatial_network_key}']['graph'] was converted from a networkx.Graph object to a pandas adjacency matrix to be saved with the AnnData object."
                warnings.warn(warn)
        mt_filename = f'{folder}/{filename}_metabolic_tasks.h5ad'
        adata.metabolic_tasks.write_h5ad(mt_filename)
        if verbose: print(f'{mt_filename} was correctly saved')
    else:
        warnings.warn('No adata.metabolic_tasks found. Skipping saving metabolic_tasks.')