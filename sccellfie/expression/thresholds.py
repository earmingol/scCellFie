import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


def get_local_percentile_threshold(adata, percentile=0.75, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    """
    Obtains the local percentile threshold for each gene in a AnnData object.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    percentile: float or list of floats, optional (default: 0.75)
        Percentile(s) to compute the threshold.

    lower_bound: float or pandas.DataFrame, optional (default: 1e-5)
        Lower bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    upper_bound: float or pandas.DataFrame, optional (default: None)
        Upper bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    exclude_zeros: bool, optional (default: False)
        Whether to exclude zeros when computing the threshold.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    thresholds: pandas.DataFrame
        A pandas.DataFrame object with the local percentile threshold for each gene.
    """
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()
    if exclude_zeros:
        X = np.where(X==0, np.nan, X)
    thresholds = np.nanquantile(X, q=percentile, axis=0, method='midpoint')
    if isinstance(percentile, list):
        columns = ['threshold-{}'.format(p) for p in percentile]
    else:
        columns = ['threshold-{}'.format(percentile)]
    thresholds = pd.DataFrame(thresholds.T, index=adata.var_names, columns=columns, dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = columns
            thresholds[thresholds < lb] = lb[thresholds < lb]
        else:
            thresholds[thresholds < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            ub.columns = columns
            thresholds[thresholds > ub] = ub[thresholds > ub]
        else:
            thresholds[thresholds > upper_bound] = upper_bound
    return thresholds


def get_global_percentile_threshold(adata, percentile=0.75, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    """
    Obtains the global percentile threshold for each gene in a AnnData object.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    percentile: float or list of floats, optional (default: 0.75)
        Percentile(s) to compute the threshold.

    lower_bound: float or pandas.DataFrame, optional (default: 1e-5)
        Lower bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    upper_bound: float or pandas.DataFrame, optional (default: None)
        Upper bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    exclude_zeros: bool, optional (default: False)
        Whether to exclude zeros when computing the threshold.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    thresholds: pandas.DataFrame
        A pandas.DataFrame object with the global percentile threshold for each gene.
    """
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()
    if exclude_zeros:
        X = np.where(X == 0, np.nan, X)
    thresholds = np.nanquantile(X, q=percentile, method='midpoint')
    if isinstance(percentile, list):
        columns = ['threshold-{}'.format(p) for p in percentile]
        thresholds = pd.DataFrame({col: [thresholds[i]]*adata.shape[1] for i, col in enumerate(columns)})
    else:
        columns = ['threshold-{}'.format(percentile)]
        thresholds = pd.DataFrame(thresholds.T, index=adata.var_names, columns=columns, dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = columns
            thresholds[thresholds < lb] = lb[thresholds < lb]
        else:
            thresholds[thresholds < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            ub.columns = columns
            thresholds[thresholds > ub] = ub[thresholds > ub]
        else:
            thresholds[thresholds > upper_bound] = upper_bound
    return thresholds


def get_local_mean_threshold(adata, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    """
    Obtains the local mean threshold for each gene in a AnnData object.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    lower_bound: float or pandas.DataFrame, optional (default: 1e-5)
        Lower bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    upper_bound: float or pandas.DataFrame, optional (default: None)
        Upper bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    exclude_zeros: bool, optional (default: False)
        Whether to exclude zeros when computing the threshold.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    thresholds: pandas.DataFrame
        A pandas.DataFrame object with the local mean threshold for each gene.
    """
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()
    if exclude_zeros:
        X = np.where(X==0, np.nan, X)
    thresholds = np.nanmean(X, axis=0)
    columns = ['threshold-mean']
    thresholds = pd.DataFrame(thresholds, index=adata.var_names, columns=columns, dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = columns
            thresholds[thresholds < lb] = lb[thresholds < lb]
        else:
            thresholds[thresholds < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            ub.columns = columns
            thresholds[thresholds > ub] = ub[thresholds > ub]
        else:
            thresholds[thresholds > upper_bound] = upper_bound
    return thresholds


def get_global_mean_threshold(adata, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    """
    Obtains the global mean threshold for each gene in a AnnData object.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    lower_bound: float or pandas.DataFrame, optional (default: 1e-5)
        Lower bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes

    upper_bound: float or pandas.DataFrame, optional (default: None)
        Upper bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes

    exclude_zeros: bool, optional (default: False)
        Whether to exclude zeros when computing the threshold.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    thresholds: pandas.DataFrame
        A pandas.DataFrame object with the global mean threshold for each gene.
    """
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()
    if exclude_zeros:
        X = np.where(X == 0, np.nan, X)
    thresholds = np.nanmean(X)
    columns = ['threshold-mean']
    thresholds = pd.DataFrame(thresholds, index=adata.var_names, columns=columns, dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = columns
            thresholds[thresholds < lb] = lb[thresholds < lb]
        else:
            thresholds[thresholds < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            ub.columns = columns
            thresholds[thresholds > ub] = ub[thresholds > ub]
        else:
            thresholds[thresholds > upper_bound] = upper_bound
    return thresholds


def get_local_trimean_threshold(adata, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    """
    Obtains the local Tukey's trimean threshold for each gene in a AnnData object.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    lower_bound: float or pandas.DataFrame, optional (default: 1e-5)
        Lower bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    upper_bound: float or pandas.DataFrame, optional (default: None)
        Upper bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    exclude_zeros: bool, optional (default: False)
        Whether to exclude zeros when computing the threshold.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    thresholds: pandas.DataFrame
        A pandas.DataFrame object with the local Tukey's trimean threshold for each gene.
    """
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()

    if exclude_zeros:
        X = np.where(X == 0, np.nan, X)
    q1 = np.nanquantile(X, q=0.25, axis=0, method='midpoint')
    median = np.nanquantile(X, q=0.5, axis=0, method='midpoint')
    q3 = np.nanquantile(X, q=0.75, axis=0, method='midpoint')
    thresholds = (q1 + 2 * median + q3) / 4

    columns = ['threshold-trimean']
    thresholds = pd.DataFrame(thresholds.T, index=adata.var_names, columns=columns, dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = columns
            thresholds[thresholds < lb] = lb[thresholds < lb]
        else:
            thresholds[thresholds < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            ub.columns = columns
            thresholds[thresholds > ub] = ub[thresholds > ub]
        else:
            thresholds[thresholds > upper_bound] = upper_bound
    return thresholds


def get_global_trimean_threshold(adata, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    """
    Obtains the global Tukey's trimean threshold for each gene in a AnnData object.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    lower_bound: float or pandas.DataFrame, optional (default: 1e-5)
        Lower bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    upper_bound: float or pandas.DataFrame, optional (default: None)
        Upper bound for the threshold. If a pandas.DataFrame is provided, it must have the same number of genes
        as the adata object.

    exclude_zeros: bool, optional (default: False)
        Whether to exclude zeros when computing the threshold.

    use_raw: bool, optional (default: False)
        Whether to use the raw data stored in adata.raw.X.

    Returns
    -------
    thresholds: pandas.DataFrame
        A pandas.DataFrame object with the global Tukey's trimean threshold for each gene.
    """
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()

    if exclude_zeros:
        X = np.where(X == 0, np.nan, X)
    q1 = np.nanquantile(X, q=0.25, method='midpoint')
    median = np.nanquantile(X, q=0.5, method='midpoint')
    q3 = np.nanquantile(X, q=0.75, method='midpoint')
    thresholds = (q1 + 2 * median + q3) / 4

    columns = ['threshold-trimean']
    thresholds = pd.DataFrame(thresholds, index=adata.var_names, columns=columns, dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = columns
            thresholds[thresholds < lb] = lb[thresholds < lb]
        else:
            thresholds[thresholds < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            ub.columns = columns
            thresholds[thresholds > ub] = ub[thresholds > ub]
        else:
            thresholds[thresholds > upper_bound] = upper_bound
    return thresholds


class _ReservoirSampler:
    """Vectorized Vitter Algorithm R — uniform sample of size `size` from an unbounded stream."""

    def __init__(self, size, rng):
        self.size = int(size)
        self.rng = rng
        self.reservoir = np.empty(self.size, dtype=np.float32)
        self.count = 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float32).ravel()
        n = values.size
        if n == 0:
            return
        if self.count < self.size:
            fill = min(self.size - self.count, n)
            self.reservoir[self.count:self.count + fill] = values[:fill]
            self.count += fill
            if fill == n:
                return
            values = values[fill:]
            n = values.size
        j = np.arange(self.count + 1, self.count + n + 1, dtype=np.int64)
        r = (self.rng.random(n) * j).astype(np.int64)
        keep = r < self.size
        if keep.any():
            self.reservoir[r[keep]] = values[keep]
        self.count += n

    def sample(self):
        return self.reservoir[:min(self.count, self.size)]


_N_COUNTS_AUTO_KEYS = ('total_counts', 'n_counts', 'raw_sum', 'nCount_RNA')


def _load_gene_set(gene_set, organism):
    """Normalize the ``gene_set`` argument into a list of gene symbols."""
    if gene_set is None:
        from sccellfie.datasets.database import load_sccellfie_database
        db = load_sccellfie_database(organism=organism)
        return list(db['thresholds'].index)
    if isinstance(gene_set, str):
        if not gene_set.lower().endswith('.json'):
            raise ValueError("String `gene_set` must point to a .json file containing a list of gene symbols.")
        with open(os.path.expanduser(gene_set)) as fp:
            loaded = json.load(fp)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a JSON list in {gene_set}, got {type(loaded).__name__}.")
        return list(loaded)
    return list(gene_set)


def _resolve_cell_index(adata, cell_mask):
    """Return a sorted integer array of selected cells."""
    n_cells = adata.n_obs
    if cell_mask is None:
        return np.arange(n_cells)
    if isinstance(cell_mask, str):
        if cell_mask not in adata.obs.columns:
            raise KeyError(f"`cell_mask='{cell_mask}'` not found in adata.obs.")
        mask = np.asarray(adata.obs[cell_mask].values).astype(bool)
    elif isinstance(cell_mask, pd.Series):
        mask = cell_mask.reindex(adata.obs_names).fillna(False).astype(bool).values
    else:
        mask = np.asarray(cell_mask)
        if mask.dtype == bool:
            if mask.shape[0] != n_cells:
                raise ValueError(f"Boolean `cell_mask` length {mask.shape[0]} != adata.n_obs {n_cells}.")
        else:
            idx = np.asarray(mask, dtype=np.int64)
            mask = np.zeros(n_cells, dtype=bool)
            mask[idx] = True
    return np.where(mask)[0]


def _source_var_names(adata, use_raw):
    if use_raw:
        if adata.raw is None:
            raise ValueError("`use_raw=True` but adata.raw is None.")
        return list(adata.raw.var_names)
    return list(adata.var_names)


def _get_chunk_matrix(chunk_adata, layer, use_raw):
    if use_raw:
        return chunk_adata.raw.X
    if layer is not None:
        return chunk_adata.layers[layer]
    return chunk_adata.X


def get_sccellfie_dataset_threshold(adata,
                                    gene_set=None,
                                    organism='human',
                                    cell_mask=None,
                                    layer=None,
                                    use_raw=False,
                                    target_sum=10_000,
                                    n_counts_key=None,
                                    chunk_size=100_000,
                                    reservoir_size=5_000_000,
                                    percentiles=(10, 25, 50, 75, 90, 95),
                                    random_state=None,
                                    verbose=True,
                                    return_stats=False):
    """
    Computes a dataset-wise ``sccellfie_threshold`` per metabolic gene by streaming
    the AnnData in chunks. Faithful port of the atlas-based threshold script that
    produced the default ``Thresholds.csv``, generalized to a single (possibly backed)
    AnnData.

    Pipeline per chunk:
      1. CP10k-normalize using a per-cell library size (obs column or computed from the full chunk).
      2. Subset to the corrected metabolic-gene columns (after applying ``CORRECT_GENES[organism]``).
      3. Accumulate per-gene sum, non-zero cell count, and max.
      4. Stream non-zero normalized values into a reservoir sample for global percentiles.

    The final threshold rule matches the original script:
        if max > P25 or max == 0:  threshold = clip(nonzero_mean, P25, P75)
        else:                      threshold = nonzero_mean

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. May be backed (``sc.read_h5ad(..., backed='r')``);
        chunks are materialized one at a time.

    gene_set : list, set, pandas.Index, str or None, optional (default: None)
        Metabolic gene list. ``None`` loads the default gene list from the scCellFie
        database for ``organism``. A string ending in ``.json`` is treated as the path
        to a JSON file containing a list of gene symbols.

    organism : str, optional (default: 'human')
        Used to select the ``CORRECT_GENES`` rename map and, if ``gene_set`` is None,
        the scCellFie database to load metabolic genes from. Currently ``'human'`` or ``'mouse'``.

    cell_mask : array-like, str or None, optional (default: None)
        Restricts the computation to a subset of cells. Accepts a boolean/integer array,
        a column name in ``adata.obs``, or a ``pandas.Series`` indexed by cell names.

    layer : str or None, optional (default: None)
        Read from ``adata.layers[layer]`` instead of ``adata.X``. Mutually exclusive with ``use_raw``.

    use_raw : bool, optional (default: False)
        Read from ``adata.raw.X``. Mutually exclusive with ``layer``.

    target_sum : float or None, optional (default: 10_000)
        Target library size for CP-normalization. Pass ``None`` to skip normalization
        (e.g. when the input values are already on the desired scale).

    n_counts_key : str or None, optional (default: None)
        Column in ``adata.obs`` containing per-cell totals. If None, auto-detect among
        ``('total_counts', 'n_counts', 'raw_sum', 'nCount_RNA')`` and otherwise compute
        per-cell sums from the full-matrix chunk before gene subsetting.

    chunk_size : int, optional (default: 100_000)
        Number of cells processed per chunk.

    reservoir_size : int, optional (default: 5_000_000)
        Size of the reservoir used to estimate global percentiles of non-zero normalized
        values. Memory cost is ``reservoir_size * 4B`` (float32).

    percentiles : tuple of int, optional (default: (10, 25, 50, 75, 90, 95))
        Percentiles to report in the returned stats. Percentiles 25 and 75 are always
        computed internally because the threshold rule depends on them.

    random_state : int or None, optional (default: None)
        Seed for the reservoir sampler.

    verbose : bool, optional (default: True)
        If True, print progress via tqdm.

    return_stats : bool, optional (default: False)
        If True, also return a dict with intermediate statistics.

    Returns
    -------
    thresholds : pandas.DataFrame
        A DataFrame indexed by metabolic gene symbol with a single column
        ``'sccellfie_threshold'``. Ready to pass to ``compute_gene_scores``
        (which selects the first column positionally).

    stats : dict, only if ``return_stats=True``
        Dict with keys ``percentiles``, ``sum_per_gene``, ``nnz_per_gene``,
        ``max_per_gene``, ``mean``, ``nonzero_mean``, ``n_cells``,
        ``n_values_seen``, ``reservoir_size_used``.
    """
    if use_raw and layer is not None:
        raise ValueError("`use_raw=True` and `layer` are mutually exclusive.")

    from sccellfie.preprocessing.prepare_inputs import CORRECT_GENES

    rename_map = CORRECT_GENES.get(organism, {})
    source_var = _source_var_names(adata, use_raw)
    corrected_var = np.array([rename_map.get(g, g) for g in source_var])

    gene_list = _load_gene_set(gene_set, organism)
    gene_set_set = set(gene_list)
    col_idx = np.where(np.array([g in gene_set_set for g in corrected_var]))[0]
    if col_idx.size == 0:
        raise ValueError("No overlap between `gene_set` and adata variables "
                         "(after applying CORRECT_GENES). Check `organism` and gene nomenclature.")
    final_gene_names = corrected_var[col_idx].tolist()

    cell_idx_full = _resolve_cell_index(adata, cell_mask)
    n_cells_sel = cell_idx_full.size
    if n_cells_sel == 0:
        raise ValueError("`cell_mask` selected zero cells.")

    already_normalized = (adata.uns.get('normalization', {}).get('method') == 'total_counts')
    do_normalize = (target_sum is not None) and (not already_normalized)

    resolved_key = None
    if do_normalize:
        if n_counts_key is not None:
            if n_counts_key not in adata.obs.columns:
                raise KeyError(f"`n_counts_key='{n_counts_key}'` not found in adata.obs.")
            resolved_key = n_counts_key
        else:
            for k in _N_COUNTS_AUTO_KEYS:
                if k in adata.obs.columns:
                    resolved_key = k
                    break

    n_genes = col_idx.size
    sum_per_gene = np.zeros(n_genes, dtype=np.float64)
    nnz_per_gene = np.zeros(n_genes, dtype=np.int64)
    max_per_gene = np.zeros(n_genes, dtype=np.float64)

    rng = np.random.default_rng(random_state)
    reservoir = _ReservoirSampler(reservoir_size, rng)

    chunk_starts = list(range(0, n_cells_sel, chunk_size))
    iterator = tqdm(chunk_starts, desc='Streaming chunks', disable=not verbose)

    for start in iterator:
        end = min(start + chunk_size, n_cells_sel)
        idx = cell_idx_full[start:end]

        chunk_adata = adata[idx]
        if getattr(adata, 'isbacked', False):
            chunk_adata = chunk_adata.to_memory()

        X_full = _get_chunk_matrix(chunk_adata, layer, use_raw)

        if do_normalize:
            if resolved_key is not None:
                n_counts_chunk = np.asarray(adata.obs[resolved_key].values[idx], dtype=np.float64)
            else:
                if sparse.issparse(X_full):
                    n_counts_chunk = np.asarray(X_full.sum(axis=1)).ravel().astype(np.float64)
                else:
                    n_counts_chunk = np.asarray(X_full.sum(axis=1), dtype=np.float64).ravel()
            safe = n_counts_chunk > 0
            scaling = np.zeros_like(n_counts_chunk)
            scaling[safe] = target_sum / n_counts_chunk[safe]
        else:
            scaling = None

        X_sub = X_full[:, col_idx]

        if scaling is not None:
            if sparse.issparse(X_sub):
                X_norm = sparse.diags(scaling, 0, format='csr') @ X_sub.tocsr()
            else:
                X_norm = np.asarray(X_sub) * scaling[:, None]
        else:
            X_norm = X_sub

        if sparse.issparse(X_norm):
            X_csr = X_norm.tocsr()
            sum_per_gene += np.asarray(X_csr.sum(axis=0)).ravel()
            nnz_per_gene += np.asarray((X_csr > 0).sum(axis=0)).ravel().astype(np.int64)
            col_max = np.asarray(X_csr.max(axis=0).todense()).ravel()
            np.maximum(max_per_gene, col_max, out=max_per_gene)
            data = X_csr.data
            nz_values = data[data > 0]
        else:
            X_dense = np.asarray(X_norm)
            sum_per_gene += X_dense.sum(axis=0)
            nnz_per_gene += (X_dense > 0).sum(axis=0).astype(np.int64)
            col_max = X_dense.max(axis=0) if X_dense.size else np.zeros(n_genes)
            np.maximum(max_per_gene, col_max, out=max_per_gene)
            nz_values = X_dense[X_dense > 0]

        reservoir.update(nz_values)

    required = {25, 75}
    all_pcts = sorted(set(int(p) for p in percentiles) | required)
    sample = reservoir.sample()
    if sample.size == 0:
        warnings.warn("No non-zero values encountered; returning zero thresholds.", UserWarning)
        pct_values = np.zeros(len(all_pcts))
    else:
        pct_values = np.percentile(sample, all_pcts)
    pct_dict = dict(zip(all_pcts, pct_values))
    p25 = float(pct_dict[25])
    p75 = float(pct_dict[75])

    nz_mean = np.zeros(n_genes, dtype=np.float64)
    has_nz = nnz_per_gene > 0
    nz_mean[has_nz] = sum_per_gene[has_nz] / nnz_per_gene[has_nz]

    clip_mask = (max_per_gene > p25) | (max_per_gene == 0)
    threshold = np.where(clip_mask, np.clip(nz_mean, p25, p75), nz_mean)

    thresholds_df = pd.DataFrame({'sccellfie_threshold': threshold},
                                 index=pd.Index(final_gene_names, name='symbol'))

    if not return_stats:
        return thresholds_df

    mean = np.zeros(n_genes, dtype=np.float64)
    if n_cells_sel > 0:
        mean = sum_per_gene / n_cells_sel
    stats = {
        'percentiles': {int(p): float(v) for p, v in pct_dict.items()},
        'sum_per_gene': pd.Series(sum_per_gene, index=final_gene_names),
        'nnz_per_gene': pd.Series(nnz_per_gene, index=final_gene_names),
        'max_per_gene': pd.Series(max_per_gene, index=final_gene_names),
        'mean': pd.Series(mean, index=final_gene_names),
        'nonzero_mean': pd.Series(nz_mean, index=final_gene_names),
        'n_cells': int(n_cells_sel),
        'n_values_seen': int(reservoir.count),
        'reservoir_size_used': int(min(reservoir.count, reservoir.size)),
    }
    return thresholds_df, stats


def set_manual_threshold(adata, threshold):
    """
    Sets a threshold manually for each gene in a AnnData object.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    threshold: float or list of floats
        Threshold(s) to be set for each gene. If a list is passed
        it must have the same number of elements as genes in adata, and
        in the same order.

    Returns
    -------
    thresholds: pandas.DataFrame
        A pandas.DataFrame object with the manual threshold for each gene.
    """
    if isinstance(threshold, list):
        assert len(adata.var_names) == len(threshold), "The len of threshold must be the same as gene number in adata"
        thresholds = pd.DataFrame(data={'threshold-manual': threshold}, index=adata.var_names, dtype=float)
    else:
        thresholds = pd.DataFrame(data={'threshold-manual': [threshold]}, index=adata.var_names, dtype=float)
    return thresholds