import pandas as pd
import numpy as np
from tqdm import tqdm
from sccellfie.expression.aggregation import agg_expression_cells


def sccellfie_to_tensor(preprocessed_db,
                        sample_key,
                        celltype_key,
                        score_type='metabolic_tasks',
                        min_cells_per_group=1,
                        agg_func='trimean',
                        layer=None,
                        gene_symbols=None,
                        top_percent=10,
                        exclude_zeros=False,
                        use_raw=False,
                        threshold=None,
                        order_labels=None,
                        sort_elements=True,
                        context_order=None,
                        fill_value=np.nan,
                        verbose=True
                        ):
    """
    Converts scCellFie scores to format compatible with cell2cell's PreBuiltTensor constructor.

    This function builds a 3D tensor with dimensions: [Contexts/Samples, Cell Types, Metabolic Features]

    Parameters
    ----------
    preprocessed_db : dict
        Output from run_sccellfie_pipeline containing 'adata' with
        metabolic_tasks and/or reactions attributes.

    sample_key : str
        Column name in adata.obs for grouping by samples/contexts.

    celltype_key : str
        Column name in adata.obs for cell type annotations.

    score_type : str, optional (default: 'metabolic_tasks')
        Which scCellFie scores to use. Options: 'metabolic_tasks', 'reactions'.

    min_cells_per_group : int, optional (default: 1)
        Minimum number of cells required per group (sample x celltype)
        to be included in analysis.

    agg_func : str, optional (default: 'trimean')
        Aggregation function to apply within cell groups. Options: 'mean', 'median',
        '25p', '75p', 'trimean', 'topmean', 'fraction_above'.

    layer : str, optional (default: None)
        Layer name to use for aggregation. If None, uses the main .X matrix.

    gene_symbols : str or list, optional (default: None)
        Specific features to include in analysis. If None, all features are used.

    top_percent : float, optional (default: 10)
        Percentage of top values for 'topmean' aggregation (0-100).

    exclude_zeros : bool, optional (default: False)
        Whether to exclude zeros when aggregating values.

    use_raw : bool, optional (default: False)
        Whether to use raw data for aggregation.

    threshold : float, optional (default: None)
        Expression threshold for 'fraction_above' aggregation.

    order_labels : list, optional (default: None)
        Labels for each dimension of the tensor. Default:
        ['Contexts', 'Cell Types', 'Metabolic Features']

    sort_elements : bool, optional (default: True)
        Whether to alphabetically sort elements in each dimension.

    context_order : list, optional (default: None)
        Custom order for contexts. If provided, contexts won't be sorted.

    fill_value : float, optional (default: numpy.nan)
        Value to fill when a feature or cell type is missing in a context.

    verbose : bool, optional (default: True)
        Whether to print information about the analysis.

    Returns
    -------
    prebuilt_tensor_args : dict
        A dictionary containing all arguments needed for PreBuiltTensor constructor:
        - 'tensor': numpy array with shape (n_contexts, n_celltypes, n_features)
        - 'order_names': list of lists with names for each dimension
        - 'order_labels': list of dimension labels
        - 'mask': mask for missing values (if applicable)
        - 'loc_nans': locations of NaN values

    Notes
    -----
    This function aggregates single-cell metabolic scores into cell type-level summaries
    across different contexts (samples, conditions, timepoints, etc.) and creates a
    tensor suitable for tensor decomposition analysis.

    The aggregation is performed using scCellFie's robust aggregation methods, which
    handle various statistical measures and can exclude zeros or use specific thresholds.

    Examples
    --------
    >>> # Convert scCellFie metabolic tasks to tensor format
    >>> tensor_args = sccellfie_to_tensor(
    ...     preprocessed_db,
    ...     sample_key='condition',
    ...     celltype_key='cell_type',
    ...     score_type='metabolic_tasks',
    ...     agg_func='mean'
    ... )
    >>>
    >>> # Create PreBuiltTensor
    >>> from cell2cell.tensor import PreBuiltTensor
    >>> tensor = PreBuiltTensor(**tensor_args)
    """

    # Extract appropriate scCellFie data
    if 'adata' not in preprocessed_db:
        raise ValueError("preprocessed_db must contain 'adata' key")

    adata = preprocessed_db['adata']

    # Select the appropriate scCellFie results
    if score_type == 'metabolic_tasks':
        if not hasattr(adata, 'metabolic_tasks'):
            raise ValueError("AnnData object must have 'metabolic_tasks' attribute. "
                             "Run scCellFie pipeline first.")
        score_adata = adata.metabolic_tasks
        feature_type = 'metabolic_task'
    elif score_type == 'reactions':
        if not hasattr(adata, 'reactions'):
            raise ValueError("AnnData object must have 'reactions' attribute. "
                             "Run scCellFie pipeline first.")
        score_adata = adata.reactions
        feature_type = 'reaction'
    else:
        raise ValueError("score_type must be either 'metabolic_tasks' or 'reactions'")

    if verbose:
        print(f"Using {score_type} with {score_adata.shape[1]} features and {score_adata.shape[0]} cells")

    # Validate required keys
    if sample_key not in score_adata.obs.columns:
        raise ValueError(f"'{sample_key}' not found in adata.obs")

    if celltype_key not in score_adata.obs.columns:
        raise ValueError(f"'{celltype_key}' not found in adata.obs")

    # Get unique elements and determine order
    samples = score_adata.obs[sample_key].unique()
    all_features = score_adata.var_names.tolist()

    # Apply gene_symbols filter if specified
    if gene_symbols is not None:
        if isinstance(gene_symbols, str):
            gene_symbols = [gene_symbols]
        features = [f for f in gene_symbols if f in all_features]
        if len(features) == 0:
            raise ValueError("None of the specified gene_symbols found in the data")
    else:
        features = all_features

    # Determine context order
    if context_order is None:
        contexts = sorted(samples) if sort_elements else list(samples)
    else:
        assert all([c in samples for c in context_order]), "context_order must contain all sample names"
        assert len(context_order) == len(samples), "Each sample must be in context_order exactly once"
        contexts = context_order

    # Aggregate data for each context and determine all cell types
    if verbose:
        print(f"Aggregating {feature_type} scores using '{agg_func}' method...")

    context_data = {}
    all_celltypes = set()

    for context in tqdm(contexts, desc='Processing contexts', disable=not verbose):
        # Subset data for this context
        context_mask = score_adata.obs[sample_key] == context
        context_adata = score_adata[context_mask, :].copy()

        # Check cell type counts
        celltype_counts = context_adata.obs[celltype_key].value_counts()
        valid_celltypes = celltype_counts[celltype_counts >= min_cells_per_group].index.tolist()

        if len(valid_celltypes) == 0:
            if verbose:
                print(f"Warning: No cell types with >= {min_cells_per_group} cells in context {context}")
            continue

        # Filter to valid cell types
        valid_mask = context_adata.obs[celltype_key].isin(valid_celltypes)
        context_adata = context_adata[valid_mask, :].copy()

        # Aggregate using scCellFie function
        try:
            agg_data = agg_expression_cells(
                adata=context_adata,
                groupby=celltype_key,
                layer=layer,
                gene_symbols=gene_symbols,
                agg_func=agg_func,
                top_percent=top_percent,
                exclude_zeros=exclude_zeros,
                use_raw=use_raw,
                threshold=threshold
            )
        except Exception as e:
            print(f"Error aggregating context {context}: {e}")
            continue

        context_data[context] = agg_data
        all_celltypes.update(agg_data.index.tolist())

    # Determine final element orders
    celltypes = sorted(all_celltypes) if sort_elements else list(all_celltypes)
    if sort_elements:
        features = sorted(features)

    if verbose:
        print(f"Building tensor with dimensions:")
        print(f"  Contexts: {len(contexts)}")
        print(f"  Cell Types: {len(celltypes)}")
        print(f"  Features: {len(features)}")

    # Build 3D tensor: [contexts, celltypes, features]
    tensor_shape = (len(contexts), len(celltypes), len(features))
    tensor = np.full(tensor_shape, fill_value, dtype=float)

    for ctx_idx, context in enumerate(tqdm(contexts, desc='Building tensor', disable=not verbose)):
        if context not in context_data:
            continue  # This context will remain filled with fill_value

        agg_data = context_data[context]

        for ct_idx, celltype in enumerate(celltypes):
            for feat_idx, feature in enumerate(features):
                # Get score for this celltype-feature combination
                if celltype in agg_data.index and feature in agg_data.columns:
                    score = agg_data.loc[celltype, feature]
                    if pd.isna(score):
                        tensor[ctx_idx, ct_idx, feat_idx] = fill_value
                    else:
                        tensor[ctx_idx, ct_idx, feat_idx] = score
                else:
                    tensor[ctx_idx, ct_idx, feat_idx] = fill_value

    # Create mask and locate NaNs
    if np.isnan(fill_value):
        mask = (~np.isnan(tensor)).astype(int)
        loc_nans = (np.isnan(tensor)).astype(int)
    else:
        mask = None
        loc_nans = np.zeros(tensor.shape, dtype=int)

    # Default order labels
    if order_labels is None:
        order_labels = ['Contexts', 'Cell Types', f'{score_type.replace("_", " ").title()}']

    # Prepare PreBuiltTensor arguments
    prebuilt_tensor_args = {
        'tensor': tensor,
        'order_names': [contexts, celltypes, features],
        'order_labels': order_labels,
        'mask': mask,
        'loc_nans': loc_nans,
    }

    if verbose:
        print(f"\nTensor built successfully!")
        print(f"Shape: {tensor.shape}")
        print(
            f"Non-zero elements: {np.count_nonzero(~np.isnan(tensor) if np.isnan(fill_value) else tensor != fill_value)}")
        print(f"Fill value: {fill_value}")

    return prebuilt_tensor_args