import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse, csr_matrix, hstack


COMPLEX_AGG_METHODS = {
    'min': lambda x, axis: np.min(x, axis=axis),
    'mean': lambda x, axis: np.mean(x, axis=axis),
    'gmean': lambda x, axis: np.exp(np.mean(np.log(np.clip(x, 1e-10, None)), axis=axis)),
}


def make_complex_name(subunits, separator='&'):
    """
    Generates a canonical complex name from a list of subunit names.

    Parameters
    ----------
    subunits : list of str
        List of subunit gene/task names.

    separator : str, default='&'
        Character(s) used to join the sorted subunit names.

    Returns
    -------
    str
        Canonical complex name with sorted subunits joined by separator.
    """
    return separator.join(sorted(subunits))


def add_complexes_to_adata(adata, complexes, agg_method='min', layer=None, copy=False):
    """
    Adds multi-gene complex expression as new variables in an AnnData object.

    Computes per-cell aggregated expression for each complex and appends
    the result as new columns in adata.X. Layers are handled by computing
    the complex aggregation for the source layer and zero-filling others.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data with individual gene expression.

    complexes : dict
        Dictionary mapping complex names (str) to lists of subunit gene names.
        Example: {'ITGA4&ITGB1': ['ITGA4', 'ITGB1']}

    agg_method : str, default='min'
        Aggregation across subunits per cell. Options:
            - 'min' : Minimum expression (rate-limiting subunit).
            - 'mean' : Arithmetic mean expression.
            - 'gmean' : Geometric mean expression.

    layer : str, optional
        Layer to read subunit expression from. If None, uses adata.X.

    copy : bool, default=False
        If True, return a modified copy. If False, modify adata in place
        and return None.

    Returns
    -------
    AnnData or None
        If copy=True, returns the modified AnnData. Otherwise modifies
        adata in place and returns None.

    Raises
    ------
    ValueError
        If agg_method is not one of 'min', 'mean', 'gmean'.
        If any subunit gene is not found in adata.var_names.
        If a complex name already exists in adata.var_names.
    """
    if agg_method not in COMPLEX_AGG_METHODS:
        raise ValueError(
            f"Invalid agg_method '{agg_method}'. Must be one of {list(COMPLEX_AGG_METHODS.keys())}"
        )

    if not complexes:
        return adata.copy() if copy else None

    # Validate subunits exist
    for complex_name, subunits in complexes.items():
        missing = [s for s in subunits if s not in adata.var_names]
        if missing:
            raise ValueError(
                f"Subunit(s) {missing} for complex '{complex_name}' not found in adata.var_names"
            )

    # Validate complex names don't collide
    existing = [name for name in complexes.keys() if name in adata.var_names]
    if existing:
        raise ValueError(
            f"Complex name(s) {existing} already exist in adata.var_names. "
            f"Remove them first or use different names."
        )

    if copy:
        adata = adata.copy()

    # Select expression source
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    is_sparse = issparse(X)

    # Compute all complex columns
    complex_cols = []
    complex_names = []
    for complex_name, subunits in complexes.items():
        subunit_indices = [adata.var_names.get_loc(s) for s in subunits]
        sub_X = X[:, subunit_indices]
        if issparse(sub_X):
            sub_X = sub_X.toarray()

        agg_values = COMPLEX_AGG_METHODS[agg_method](sub_X, axis=1)
        complex_cols.append(agg_values.reshape(-1, 1))
        complex_names.append(complex_name)

    complex_matrix = np.hstack(complex_cols)  # (n_cells, n_complexes)

    # Build new X via hstack
    if is_sparse:
        new_X = hstack([adata.X, csr_matrix(complex_matrix)], format='csr')
    else:
        new_X = np.hstack([adata.X, complex_matrix])

    # Build new var with metadata
    new_var_entries = pd.DataFrame(
        {
            'is_complex': [True] * len(complex_names),
            'complex_subunits': ['|'.join(complexes[n]) for n in complex_names],
            'complex_agg_method': [agg_method] * len(complex_names),
        },
        index=complex_names,
    )

    old_var = adata.var.copy()
    if 'is_complex' not in old_var.columns:
        old_var['is_complex'] = False
        old_var['complex_subunits'] = np.nan
        old_var['complex_agg_method'] = np.nan

    new_var = pd.concat([old_var, new_var_entries])

    # Handle layers
    new_layers = {}
    if adata.layers is not None:
        for layer_name, layer_data in adata.layers.items():
            layer_is_sparse = issparse(layer_data)

            if layer_name == layer:
                # Source layer: compute complex aggregation
                layer_complex_cols = []
                for complex_name, subunits in complexes.items():
                    subunit_indices = [adata.var_names.get_loc(s) for s in subunits]
                    sub_L = layer_data[:, subunit_indices]
                    if issparse(sub_L):
                        sub_L = sub_L.toarray()
                    agg_vals = COMPLEX_AGG_METHODS[agg_method](sub_L, axis=1)
                    layer_complex_cols.append(agg_vals.reshape(-1, 1))

                layer_complex_matrix = np.hstack(layer_complex_cols)

                if layer_is_sparse:
                    new_layers[layer_name] = hstack(
                        [layer_data, csr_matrix(layer_complex_matrix)], format='csr'
                    )
                else:
                    new_layers[layer_name] = np.hstack([layer_data, layer_complex_matrix])
            else:
                # Non-source layer: zero-fill
                n_new = len(complex_names)
                if layer_is_sparse:
                    zero_cols = csr_matrix((layer_data.shape[0], n_new))
                    new_layers[layer_name] = hstack([layer_data, zero_cols], format='csr')
                else:
                    zero_cols = np.zeros((layer_data.shape[0], n_new))
                    new_layers[layer_name] = np.hstack([layer_data, zero_cols])

    # Preserve obs with categorical dtypes
    new_obs = adata.obs.copy()
    for col in new_obs.columns:
        if pd.api.types.is_categorical_dtype(adata.obs[col]):
            new_obs[col] = new_obs[col].astype('category')
            if hasattr(adata.obs[col].cat, 'ordered') and adata.obs[col].cat.ordered:
                new_obs[col] = new_obs[col].cat.reorder_categories(adata.obs[col].cat.categories)
                new_obs[col] = new_obs[col].cat.as_ordered()

    # Reconstruct AnnData
    new_obsm = adata.obsm.copy() if adata.obsm is not None else None
    new_obsp = adata.obsp.copy() if adata.obsp is not None else None

    adata_new = sc.AnnData(
        X=new_X,
        obs=new_obs,
        var=new_var,
        uns=adata.uns.copy() if adata.uns else None,
        obsm=new_obsm,
        obsp=new_obsp,
        layers=new_layers if new_layers else None,
    )

    if copy:
        return adata_new
    else:
        adata.__dict__.update(adata_new.__dict__)
        return None


def prepare_var_pairs(adata, var_pairs, complex_sep='&', agg_method='min', layer=None):
    """
    Prepares variable pairs for communication scoring by detecting
    multi-element (complex) entries, adding them to adata, and returning
    normalized string-only pairs.

    Each element in a var_pair can be either a string (single gene/task)
    or a list/tuple of strings (complex with multiple subunits). When a
    list is detected, the complex is automatically named by joining the
    sorted subunit names with complex_sep and added to adata via
    add_complexes_to_adata. Complexes already present in adata.var_names
    are skipped.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data.

    var_pairs : list of tuples
        List of (ligand, receptor) pairs where each element can be:
            - str: single gene or task name.
            - list/tuple of str: subunits of a complex.

        Example::

            var_pairs = [
                (['TASK1', 'TASK2'], ['GENE1', 'GENE2']),  # both complex
                ('TASK3', 'GENE4'),                          # both single
                ('TASK1', ['GENE5', 'GENE6']),               # mixed
            ]

    complex_sep : str, default='&'
        Separator used to join subunit names into the complex name.

    agg_method : str, default='min'
        Aggregation method for complex subunits. See add_complexes_to_adata.

    layer : str, optional
        Layer to read subunit expression from.

    Returns
    -------
    normalized_pairs : list of tuples
        String-only (ligand, receptor) pairs ready for scoring functions.
        Complex elements are replaced by their generated names.
    """
    complexes_to_add = {}
    normalized_pairs = []

    for var1, var2 in var_pairs:
        # Normalize each element
        if isinstance(var1, (list, tuple)):
            name1 = make_complex_name(var1, separator=complex_sep)
            if name1 not in adata.var_names and name1 not in complexes_to_add:
                complexes_to_add[name1] = list(var1)
        else:
            name1 = var1

        if isinstance(var2, (list, tuple)):
            name2 = make_complex_name(var2, separator=complex_sep)
            if name2 not in adata.var_names and name2 not in complexes_to_add:
                complexes_to_add[name2] = list(var2)
        else:
            name2 = var2

        normalized_pairs.append((name1, name2))

    if complexes_to_add:
        add_complexes_to_adata(adata, complexes_to_add, agg_method=agg_method, layer=layer)

    return normalized_pairs
