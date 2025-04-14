import os
import warnings

import networkx as nx
import pandas as pd
from pathlib import Path


def save_adata(adata, output_directory, filename, spatial_network_key='spatial_network', verbose=True):
    """
    Saves an AnnData object and its scCellFie attributes to a folder.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    output_directory: str
        Directory to save the results (AnnData objects).

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
    """
    # Check folder path
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    if spatial_network_key in adata.uns.keys():
        if isinstance(adata.uns[spatial_network_key]['graph'], nx.Graph):
            adata.uns[spatial_network_key]['graph'] = nx.to_pandas_adjacency(adata.uns[spatial_network_key]['graph'])
            warn = f"adata.uns['{spatial_network_key}']['graph'] was converted from a networkx.Graph object to a pandas adjacency matrix to be saved with the AnnData object."
            warnings.warn(warn)

    adata_filename = f'{output_directory}/{filename}.h5ad'
    adata.write_h5ad(adata_filename)
    if verbose: print(f'{adata_filename} was correctly saved')

    if hasattr(adata, 'reactions'):
        if spatial_network_key in adata.reactions.uns.keys():
            if isinstance(adata.reactions.uns[spatial_network_key]['graph'], nx.Graph):
                adata.reactions.uns[spatial_network_key]['graph'] = nx.to_pandas_adjacency(adata.reactions.uns[spatial_network_key]['graph'])
                warn = f"adata.reactions.uns['{spatial_network_key}']['graph'] was converted from a networkx.Graph object to a pandas adjacency matrix to be saved with the AnnData object."
                warnings.warn(warn)
        reaction_filename = f'{output_directory}/{filename}_reactions.h5ad'
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
        mt_filename = f'{output_directory}/{filename}_metabolic_tasks.h5ad'
        adata.metabolic_tasks.write_h5ad(mt_filename)
        if verbose: print(f'{mt_filename} was correctly saved')
    else:
        warnings.warn('No adata.metabolic_tasks found. Skipping saving metabolic_tasks.')


def save_result_summary(results_dict, output_directory, prefix=''):
    """
    Save the result summary contained in a dictionary to CSV files.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing the DataFrames with results from the
        sccellfie.reports.summary.generate_report_from_adata() function.

    output_directory : str
        Directory to save the results.

    prefix : str, optional (default: '')
        Prefix to add to the filenames.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Add prefix if provided
    if prefix and not prefix.endswith('-'):
        prefix = f"{prefix}-"

    # Save each DataFrame
    for key, df in results_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        filename = f"{prefix}{key.capitalize()}.csv"
        filepath = os.path.join(output_directory, filename)

        # Determine if index should be included
        save_index = key not in ['cell_counts', 'melted']

        df.to_csv(filepath, index=save_index)

    print(f"Results saved to {output_directory}")