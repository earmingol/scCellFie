import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from pandas.core.algorithms import duplicated
from scipy import sparse
from scipy.sparse import issparse, csr_matrix, hstack

from sccellfie.datasets.gene_info import retrieve_ensembl2symbol_data
from sccellfie.preprocessing.matrix_utils import get_matrix_gene_expression


def get_adata_gene_expression(adata, gene, layer=None, use_raw=False):
    """
    Get expression values for a given feature from AnnData object.
    Checks both adata.var_names (gene expression) and adata.obs (metadata).

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the expression data.

    gene : str
        Name of the gene or feature to extract the expression values for.

    layer : str, optional (default: None)
        Name of the layer to extract the expression values from.
        This layer has priority over adata.X and ´use_raw´.

    use_raw : bool, optional (default: False)
        If True, use the raw data in adata.raw.X if available.

    Returns
    -------
    expression : numpy.ndarray
        Array containing the expression values for the specified gene.
    """
    if layer is not None:
        X = adata.layers[layer]
    elif (use_raw) and (adata.raw is not None):
        X = adata.raw.X
    else:
        X = adata.X

    if gene in adata.var_names:
        expression = get_matrix_gene_expression(X, adata.var_names, gene)
    elif gene in adata.obs.columns:
        # Extract values from observation metadata
        expression = adata.obs[gene].values

        # Convert to numeric if possible
        try:
            expression = expression.astype(float)
        except ValueError:
            raise ValueError(f"Feature '{gene}' in adata.obs is not numeric")
    else:
        raise ValueError(f"Feature '{gene}' not found in either adata.var_names or adata.obs columns")

    return expression


def stratified_subsample_adata(adata, group_column, target_fraction=0.20, random_state=0):
    """
    Stratified subsampling of an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    group_column : str
        Column name in adata.obs containing the group information.

    target_fraction : float, optional (default: 0.20)
        Fraction of cells to sample from each group.

    random_state : int, optional (default: 0)
        Random seed for reproducibility.

    Returns
    -------
    adata_subsampled : AnnData
        Subsampled AnnData object
    """
    # Get the indices of the cells to keep
    indices_to_keep = (adata.obs
                       .groupby(group_column)
                       .apply(lambda x: x.sample(frac=target_fraction, random_state=random_state))
                       .index.get_level_values(1))

    # Create a new AnnData object with the subsampled cells
    adata_subsampled = adata[indices_to_keep]
    return adata_subsampled


def normalize_adata(adata, target_sum=10_000, n_counts_key='n_counts', copy=False):
    """
    Preprocesses an AnnData object by normalizing the data to a target sum.
    Original adata object is updated in place.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the expression data.

    target_sum : int, optional (default: 10_000)
        The target sum to which the data will be normalized.

    n_counts_key : str, optional (default: 'n_counts')
        The key in adata.obs containing the total counts for each cell.

    copy : bool, optional (default: False)
        If True, returns a copy of adata with the normalized data.
    """
    if copy:
        adata = adata.copy()

    # Check if total counts are already calculated
    if n_counts_key not in adata.obs.columns:
        warnings.warn(f"{n_counts_key} not found in adata.obs. Calculating total counts.", UserWarning)
        n_counts_key = 'total_counts'  # scanpy uses 'total_counts' as the key
        # Calculate total counts from the raw expression matrix
        adata.obs[n_counts_key] = adata.X.sum(axis=1)

    # Input data
    X_view = adata.X

    warnings.warn("Normalizing data.", UserWarning)

    # Check if matrix is sparse
    is_sparse = sparse.issparse(X_view)

    # Convert to dense if sparse
    if is_sparse:
        X_view = X_view.toarray()

    # Normalize
    n_counts = adata.obs[n_counts_key].values[:, None]
    X_norm = X_view / n_counts * target_sum

    # Convert back to sparse if original was sparse
    if is_sparse:
        X_norm = sparse.csr_matrix(X_norm)

    # Update adata
    adata.X = X_norm
    adata.uns['normalization'] = {
        'method': 'total_counts',
        'target_sum': target_sum,
        'n_counts_key': n_counts_key
    }
    if copy:
        return adata


def transform_adata_gene_names(adata, filename=None, organism='human', copy=True, drop_unmapped=False):
    """
    Transforms gene names in an AnnData object from Ensembl IDs to gene symbols.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the expression data. All gene names
        must be in Ensembl ID format.

    filename : str, optional
        The file path to a custom CSV file containing Ensembl IDs and gene symbols.
        One column must be 'ensembl_id' and the other 'symbol'.

    organism : str, optional (default: 'human')
        The organism to retrieve data for. Choose 'human' or 'mouse'.

    copy : bool, optional (default: True)
        If True, return a copy of the AnnData object. If False, modify the object in place.

    drop_unmapped : bool, optional (default: False)
        If True, drop genes that could not be mapped to symbols.

    Returns
    -------
    AnnData
        The AnnData object with gene names transformed to gene symbols.
        If copy=True, this is a new object.

    Raises
    ------
    ValueError
        If not all genes in the AnnData object are in Ensembl ID format.
    """
    # Retrieve the Ensembl ID to gene symbol mapping
    ensembl2symbol = retrieve_ensembl2symbol_data(filename, organism)

    if not ensembl2symbol:
        raise ValueError("Failed to retrieve Ensembl ID to gene symbol mapping.")

    # Check if all genes are in Ensembl format
    all_ensembl = all(gene.startswith('ENS') for gene in adata.var_names)
    if not all_ensembl:
        raise ValueError("Not all genes are in Ensembl ID format. Please ensure all genes start with 'ENS'.")

    # Create a new AnnData object if copy is True, otherwise use the original
    adata_mod = adata.copy() if copy else adata

    # Create a mapping Series
    gene_map = pd.Series(ensembl2symbol)

    # Transform gene names
    new_var_names = adata_mod.var_names.map(gene_map)

    # Check if any genes were not mapped
    unmapped = new_var_names.isna()
    if unmapped.any():
        unmapped_count = unmapped.sum()
        print(f"Warning: {unmapped_count} genes could not be mapped to symbols.")

        if drop_unmapped:
            print(f"Dropping {unmapped_count} unmapped genes.")
            adata_mod = adata_mod[:, ~unmapped].copy()
            new_var_names = new_var_names[~unmapped]
        else:
            # For unmapped genes, keep the original Ensembl ID
            new_var_names = pd.Index([new_name if pd.notna(new_name) else old_name
                                      for new_name, old_name in zip(new_var_names, adata_mod.var_names)])

    # Assign new gene names
    adata_mod.var_names = new_var_names

    # Drop duplicates
    duplicated = adata_mod.var_names.duplicated(keep='first').any()
    if duplicated:
        print("Duplicated gene names found. Dropping duplicates.")
        mask = ~adata_mod.var_names.duplicated(keep='first')
        adata_mod = adata_mod[:, mask]
    return adata_mod


def transfer_variables(adata_target, adata_source, var_names, source_obs_col=None, target_obs_col=None,
                       keep_sparse=True):
    """
    Transfers variables from source AnnData to target AnnData, handling different sizes
    and maintaining sparse matrix format if needed.

    Parameters
    ----------
    adata_target : AnnData
        Target AnnData object to add variables to.

    adata_source : AnnData
        Source AnnData object to get variables from.

    var_names : str or list
        Names of variables to transfer from ´adata_source´ to ´adata_target´.

    source_obs_col : str, optional
        Column in source adata.obs to use for matching observations
        (e.g. column containing barcodes).

    target_obs_col : str, optional
        Column in target adata.obs to use for matching observations
        (e.g. column containing barcodes).

    keep_sparse : bool
        Whether to maintain sparse matrix format if present.

    Returns
    -------
    AnnData
        Updated target AnnData object with new variables
    """
    # Convert var_names to list if string
    if isinstance(var_names, str):
        var_names = [var_names]

    # Verify variables exist in source
    missing_vars = [v for v in var_names if v not in adata_source.var_names]
    if missing_vars:
        raise ValueError(f"Variables not found in source: {missing_vars}")

    # Verify variables don't already exist in target
    existing_vars = [v for v in var_names if v in adata_target.var_names]
    if existing_vars:
        warnings.warn(f"Variables already exist in target. These will not be transferred: {existing_vars}")
        var_names = [v for v in var_names if v not in existing_vars]

    # Create new var DataFrame
    new_var = adata_target.var.copy()

    # Add a temporary column to handle empty DataFrame cases
    temp_col_name = '_temp_transfer_column_'
    new_var[temp_col_name] = np.nan

    # Add temporary column to source var copy as well
    source_var = adata_source.var.copy()
    source_var[temp_col_name] = np.nan

    # Original column processing logic
    cols_in_both = new_var.columns.intersection(source_var.columns)
    cols_only_source = source_var.columns.difference(new_var.columns)
    new_cols = cols_in_both.union(cols_only_source)

    if len(new_cols) > 0:
        new_var = new_var[cols_in_both]
        new_var = new_var.reindex(columns=new_cols)
        source_var = source_var[new_cols]

    for v in var_names:
        if v in source_var.index:
            new_var.loc[v] = source_var.loc[v]
        else:
            new_var.loc[v] = None

    # If matching columns are provided, use them to align observations
    if source_obs_col is not None and target_obs_col is not None:
        # Get indices for matching observations
        source_ids = adata_source.obs[source_obs_col]
        target_ids = adata_target.obs[target_obs_col]

        # Create mapping from source to target indices
        id_map = {id_: idx for idx, id_ in enumerate(source_ids)}
        target_to_source = np.array([id_map.get(id_, -1) for id_ in target_ids])

        # Check if any observations couldn't be mapped
        if np.any(target_to_source == -1):
            raise ValueError("Some observations in target could not be mapped to source")

        # Extract data for matching observations
        new_data = adata_source[target_to_source, var_names].X
    else:
        # Without matching columns, require same number of observations
        if adata_source.n_obs != adata_target.n_obs:
            raise ValueError(
                f"Number of observations don't match: {adata_source.n_obs} vs {adata_target.n_obs}"
            )
        new_data = adata_source[:, var_names].X

    # Ensure sparse matrix format if needed
    if keep_sparse and not issparse(new_data):
        new_data = csr_matrix(new_data)
    elif not keep_sparse and issparse(new_data):
        new_data = new_data.toarray()

    # Create new AnnData object with updated dimensions
    if keep_sparse:
        if not issparse(adata_target.X):
            X = csr_matrix(adata_target.X)
        else:
            X = adata_target.X
        combined_X = hstack([X, new_data])
    else:
        if issparse(adata_target.X):
            X = adata_target.X.toarray()
        else:
            X = adata_target.X
        combined_X = np.hstack([X, new_data])

    # Handle layers
    new_layers = {}
    if adata_target.layers is not None:
        for layer_name, layer_data in adata_target.layers.items():
            # Check if this layer exists in source
            if adata_source.layers is not None and layer_name in adata_source.layers:
                # Get layer data for new variables from source
                if source_obs_col is not None and target_obs_col is not None:
                    new_layer_data = adata_source[target_to_source, var_names].layers[layer_name]
                else:
                    new_layer_data = adata_source[:, var_names].layers[layer_name]

                # Ensure correct format
                if keep_sparse:
                    if not issparse(layer_data):
                        layer_matrix = csr_matrix(layer_data)
                    else:
                        layer_matrix = layer_data
                    if not issparse(new_layer_data):
                        new_layer_data = csr_matrix(new_layer_data)
                    new_layers[layer_name] = hstack([layer_matrix, new_layer_data])
                else:
                    if issparse(layer_data):
                        layer_matrix = layer_data.toarray()
                    else:
                        layer_matrix = layer_data
                    if issparse(new_layer_data):
                        new_layer_data = new_layer_data.toarray()
                    new_layers[layer_name] = np.hstack([layer_matrix, new_layer_data])
            else:
                # Layer doesn't exist in source, add zeros for new variables
                if keep_sparse:
                    if not issparse(layer_data):
                        layer_matrix = csr_matrix(layer_data)
                    else:
                        layer_matrix = layer_data
                    zero_cols = csr_matrix((layer_matrix.shape[0], len(var_names)))
                    new_layers[layer_name] = hstack([layer_matrix, zero_cols])
                else:
                    if issparse(layer_data):
                        layer_matrix = layer_data.toarray()
                    else:
                        layer_matrix = layer_data
                    zero_cols = np.zeros((layer_matrix.shape[0], len(var_names)))
                    new_layers[layer_name] = np.hstack([layer_matrix, zero_cols])

    # Handle obsm - combine target and source
    new_obsm = adata_target.obsm.copy() if adata_target.obsm is not None else {}
    if adata_source.obsm is not None:
        # For source obsm data, we need to handle the same observation alignment as we did for X
        for key in adata_source.obsm.keys():
            if key not in new_obsm:  # Only add keys that don't exist in target
                if source_obs_col is not None and target_obs_col is not None:
                    new_obsm[key] = adata_source.obsm[key][target_to_source]
                else:
                    new_obsm[key] = adata_source.obsm[key]

    # Handle obsp - combine target and source
    new_obsp = adata_target.obsp.copy() if adata_target.obsp is not None else {}
    if adata_source.obsp is not None:
        for key in adata_source.obsp.keys():
            if key not in new_obsp:  # Only add keys that don't exist in target
                if source_obs_col is not None and target_obs_col is not None:
                    new_obsp[key] = adata_source.obsp[key][target_to_source][:, target_to_source]
                else:
                    new_obsp[key] = adata_source.obsp[key]

    # Create new obs with preserved dtypes
    new_obs = adata_target.obs.copy()

    # Preserve categorical dtypes
    for col in new_obs.columns:
        if pd.api.types.is_categorical_dtype(adata_target.obs[col]):
            new_obs[col] = new_obs[col].astype('category')
            # Preserve category ordering if it exists
            if hasattr(adata_target.obs[col], 'cat') and hasattr(adata_target.obs[col].cat, 'ordered') and \
                    adata_target.obs[col].cat.ordered:
                new_obs[col] = new_obs[col].cat.reorder_categories(adata_target.obs[col].cat.categories)
                new_obs[col] = new_obs[col].cat.as_ordered()

    # Remove the temporary column before creating the new AnnData
    if temp_col_name in new_var.columns:
        new_var = new_var.drop(columns=[temp_col_name])

    # Create new AnnData with correct dimensions
    adata_new = sc.AnnData(
        X=combined_X,
        obs=new_obs,
        var=new_var,
        uns=adata_target.uns.copy(),
        obsm=new_obsm if new_obsm else None,
        obsp=new_obsp if new_obsp else None,
        varm=adata_target.varm.copy() if adata_target.varm is not None else None,
        layers=new_layers if new_layers else None
    )

    return adata_new