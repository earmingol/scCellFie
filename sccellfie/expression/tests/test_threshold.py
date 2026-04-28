import json
import os

import numpy as np
import pytest
import pandas as pd
import scanpy as sc
from scipy import sparse

from pandas.testing import assert_frame_equal
from sccellfie.expression.thresholds import (get_local_mean_threshold, get_global_mean_threshold,
                                             get_local_percentile_threshold, get_global_percentile_threshold,
                                             get_local_trimean_threshold, get_global_trimean_threshold,
                                             get_sccellfie_dataset_threshold, set_manual_threshold,
                                             _ReservoirSampler)
from sccellfie.datasets.toy_inputs import create_controlled_adata


@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-0.75': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-0.75': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_local_percentile_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    percentile = 0.75
    thresholds = get_local_percentile_threshold(adata,
                                                percentile=percentile,
                                                use_raw=use_raw,
                                                lower_bound=lower_bound,
                                                upper_bound=upper_bound,
                                                exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({f'threshold-{percentile}': [6, 7, 8]},
                                  index=['gene1', 'gene2', 'gene3'],
                                  dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = [f'threshold-{percentile}']
            expected_values[expected_values < lb] = lb[expected_values < lb]
        else:
            expected_values[expected_values < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]
        else:
            expected_values[expected_values > upper_bound] = upper_bound

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-{}'.format(percentile) in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert thresholds.equals(expected_values), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert all(thresholds >= lower_bound), "Lower bound not respected"
    if upper_bound is not None:
        assert all(thresholds <= upper_bound), "Upper bound not respected"


@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-0.75': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-0.75': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_global_percentile_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    percentile = 0.75
    thresholds = get_global_percentile_threshold(adata,
                                                 percentile=percentile,
                                                 use_raw=use_raw,
                                                 lower_bound=lower_bound,
                                                 upper_bound=upper_bound,
                                                 exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({f'threshold-{percentile}': [6.5, 6.5, 6.5]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            lb.columns = [f'threshold-{percentile}']
            expected_values[expected_values < lb] = lb[expected_values < lb]
        else:
            expected_values[expected_values < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]
        else:
            expected_values[expected_values > upper_bound] = upper_bound

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-{}'.format(percentile) in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert thresholds.equals(expected_values), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert all(thresholds >= lower_bound), "Lower bound not respected"
    if upper_bound is not None:
        assert all(thresholds <= upper_bound), "Upper bound not respected"


@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-mean': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-mean': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_local_mean_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    thresholds = get_local_mean_threshold(adata,
                                          use_raw=use_raw,
                                          lower_bound=lower_bound,
                                          upper_bound=upper_bound,
                                          exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({'threshold-mean': [4. , 5. , 4.5]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    if exclude_zeros:
        expected_values.loc['gene3', 'threshold-mean'] = 6.0

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            expected_values[expected_values < lb] = lb[expected_values < lb]
        else:
            expected_values[expected_values < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]
        else:
            expected_values[expected_values > upper_bound] = upper_bound

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-mean' in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert thresholds.equals(expected_values), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert all(thresholds >= lower_bound), "Lower bound not respected"
    if upper_bound is not None:
        assert all(thresholds <= upper_bound), "Upper bound not respected"

@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-mean': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-mean': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_global_mean_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    thresholds = get_global_mean_threshold(adata,
                                           use_raw=use_raw,
                                           lower_bound=lower_bound,
                                           upper_bound=upper_bound,
                                           exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({'threshold-mean': [4.5 , 4.5 , 4.5]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    if exclude_zeros:
        expected_values = pd.DataFrame({'threshold-mean': [4.909091, 4.909091, 4.909091]},
                                       index=['gene1', 'gene2', 'gene3'],
                                       dtype=float)

    if lower_bound is not None:
        if type(lower_bound) not in (int, float, complex):
            lb = lower_bound.copy()
            expected_values[expected_values < lb] = lb[expected_values < lb]
        else:
            expected_values[expected_values < lower_bound] = lower_bound

    if upper_bound is not None:
        if type(upper_bound) not in (int, float, complex):
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]
        else:
            expected_values[expected_values > upper_bound] = upper_bound

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-mean' in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    print(thresholds, expected_values)
    assert_frame_equal(thresholds, expected_values, check_exact=False), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert all(thresholds >= lower_bound), "Lower bound not respected"
    if upper_bound is not None:
        assert all(thresholds <= upper_bound), "Upper bound not respected"


def test_percentile_list():
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    percentile = [0.25, 0.50, 0.75]
    local_thresholds = get_local_percentile_threshold(adata, percentile=percentile)
    global_thresholds = get_global_percentile_threshold(adata, percentile=percentile)

    # Expected columns
    columns = ['threshold-{}'.format(p) for p in percentile]

    # Test output structure
    assert all(col in local_thresholds.columns for col in columns), "Missing columns"
    assert all(col in global_thresholds.columns for col in columns), "Missing columns"


@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-trimean': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-trimean': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_local_trimean_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    thresholds = get_local_trimean_threshold(adata,
                                             use_raw=use_raw,
                                             lower_bound=lower_bound,
                                             upper_bound=upper_bound,
                                             exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({'threshold-trimean': [4., 5., 4.25]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    if exclude_zeros:
        expected_values.loc['gene3', 'threshold-trimean'] = 6.0

    if lower_bound is not None:
        if isinstance(lower_bound, (int, float, complex)):
            expected_values[expected_values < lower_bound] = lower_bound
        else:
            lb = lower_bound.copy()
            expected_values[expected_values < lb] = lb[expected_values < lb]

    if upper_bound is not None:
        if isinstance(upper_bound, (int, float, complex)):
            expected_values[expected_values > upper_bound] = upper_bound
        else:
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-trimean' in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert_frame_equal(thresholds, expected_values, check_exact=False), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert (thresholds >= lower_bound).all().all(), "Lower bound not respected"
    if upper_bound is not None:
        assert (thresholds <= upper_bound).all().all(), "Upper bound not respected"


@pytest.mark.parametrize("use_raw, lower_bound, upper_bound, exclude_zeros",
                         [(False, 1e-5, None, False),
                          (True, 1e-5, None, False),
                          (False, 10, None, False),
                          (False, 1e-5, 4, False),
                          (False, 1e-5, None, True),
                          (False, pd.DataFrame({f'threshold-trimean': [10, 8, 12]}, index=['gene1', 'gene2', 'gene3'], dtype=float), None, False),
                          (False, 1e-5, pd.DataFrame({f'threshold-trimean': [1, 2, 3]}, index=['gene1', 'gene2', 'gene3'], dtype=float), False),
                         ])
def test_get_global_trimean_threshold(use_raw, lower_bound, upper_bound, exclude_zeros):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    thresholds = get_global_trimean_threshold(adata,
                                              use_raw=use_raw,
                                              lower_bound=lower_bound,
                                              upper_bound=upper_bound,
                                              exclude_zeros=exclude_zeros)

    # Expected values
    expected_values = pd.DataFrame({'threshold-trimean': [4.375, 4.375, 4.375]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    if exclude_zeros:
        expected_values = pd.DataFrame({'threshold-trimean': [4.75, 4.75, 4.75]},
                                       index=['gene1', 'gene2', 'gene3'],
                                       dtype=float)

    if lower_bound is not None:
        if isinstance(lower_bound, (int, float, complex)):
            expected_values[expected_values < lower_bound] = lower_bound
        else:
            lb = lower_bound.copy()
            expected_values[expected_values < lb] = lb[expected_values < lb]

    if upper_bound is not None:
        if isinstance(upper_bound, (int, float, complex)):
            expected_values[expected_values > upper_bound] = upper_bound
        else:
            ub = upper_bound.copy()
            expected_values[expected_values > ub] = ub[expected_values > ub]

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-trimean' in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert_frame_equal(thresholds, expected_values, check_exact=False), "Threshold values do not match expected results"
    if lower_bound is not None:
        assert (thresholds >= lower_bound).all().all(), "Lower bound not respected"
    if upper_bound is not None:
        assert (thresholds <= upper_bound).all().all(), "Upper bound not respected"


def test_set_manual_threshold():
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    thresholds = set_manual_threshold(adata, threshold=5)

    # Expected values
    expected_values = pd.DataFrame({'threshold-manual': [5, 5, 5]},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-manual' in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert thresholds.equals(expected_values), "Threshold values do not match expected results"


def test_manual_threshold_per_gene():
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Run the function
    threshold_list = [5, 10, 15]
    thresholds = set_manual_threshold(adata, threshold=threshold_list)

    # Expected values
    expected_values = pd.DataFrame({'threshold-manual': threshold_list},
                                   index=['gene1', 'gene2', 'gene3'],
                                   dtype=float)

    # Test output structure
    assert isinstance(thresholds, pd.DataFrame)
    assert 'threshold-manual' in thresholds.columns
    assert thresholds.shape == expected_values.shape

    # Test correctness of values
    assert thresholds.equals(expected_values), "Threshold values do not match expected results"


def _reference_dataset_threshold(X, var_names, gene_set, target_sum=10_000, normalize=True):
    """Naive in-memory reference for get_sccellfie_dataset_threshold."""
    X = (X.toarray() if sparse.issparse(X) else np.asarray(X)).astype(np.float64)
    if normalize:
        n_counts = X.sum(axis=1, keepdims=True)
        scaling = np.zeros_like(n_counts, dtype=np.float64)
        mask = n_counts[:, 0] > 0
        scaling[mask, 0] = target_sum / n_counts[mask, 0]
        X_norm = X * scaling
    else:
        X_norm = X
    keep = [i for i, g in enumerate(var_names) if g in set(gene_set)]
    names = [var_names[i] for i in keep]
    X_sub = X_norm[:, keep]
    sum_nz = X_sub.sum(axis=0)
    nnz = (X_sub > 0).sum(axis=0)
    max_vals = X_sub.max(axis=0) if X_sub.size else np.zeros(len(names))
    nz_mean = np.where(nnz > 0, sum_nz / np.maximum(nnz, 1), 0.0)
    all_nz = X_sub[X_sub > 0]
    if all_nz.size == 0:
        p25 = p75 = 0.0
    else:
        p25, p75 = np.percentile(all_nz, [25, 75])
    clip_mask = (max_vals > p25) | (max_vals == 0)
    threshold = np.where(clip_mask, np.clip(nz_mean, p25, p75), nz_mean)
    return pd.DataFrame({'sccellfie_threshold': threshold},
                        index=pd.Index(names, name='symbol'))


def test_reservoir_sampler_fill_only():
    rng = np.random.default_rng(0)
    sampler = _ReservoirSampler(size=10, rng=rng)
    sampler.update(np.array([1.0, 2.0, 3.0]))
    out = sampler.sample()
    assert out.size == 3
    np.testing.assert_array_equal(np.sort(out), np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_reservoir_sampler_exact_when_size_exceeds_stream():
    rng = np.random.default_rng(42)
    sampler = _ReservoirSampler(size=100, rng=rng)
    for batch in np.array_split(np.arange(1, 21, dtype=np.float32), 4):
        sampler.update(batch)
    out = np.sort(sampler.sample())
    np.testing.assert_array_equal(out, np.arange(1, 21, dtype=np.float32))


def test_reservoir_sampler_deterministic():
    values = np.random.default_rng(123).random(5000).astype(np.float32)
    s1 = _ReservoirSampler(size=500, rng=np.random.default_rng(7))
    s2 = _ReservoirSampler(size=500, rng=np.random.default_rng(7))
    s1.update(values)
    s2.update(values)
    np.testing.assert_array_equal(s1.sample(), s2.sample())


def test_get_sccellfie_dataset_threshold_matches_reference():
    adata = create_controlled_adata()
    gene_set = ['gene1', 'gene2', 'gene3']
    expected = _reference_dataset_threshold(adata.X, list(adata.var_names), gene_set)
    out = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, chunk_size=100,
        reservoir_size=100, random_state=0, verbose=False,
    )
    assert list(out.columns) == ['sccellfie_threshold']
    assert list(out.index) == ['gene1', 'gene2', 'gene3']
    np.testing.assert_allclose(out['sccellfie_threshold'].values,
                               expected['sccellfie_threshold'].values, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 10])
def test_get_sccellfie_dataset_threshold_chunking_exact_when_reservoir_covers_stream(chunk_size):
    adata = create_controlled_adata()
    gene_set = ['gene1', 'gene2', 'gene3']
    reference = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, chunk_size=100,
        reservoir_size=100, random_state=0, verbose=False,
    )
    out = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, chunk_size=chunk_size,
        reservoir_size=100, random_state=0, verbose=False,
    )
    assert_frame_equal(out, reference)


def test_get_sccellfie_dataset_threshold_sparse_vs_dense():
    adata = create_controlled_adata()
    dense = adata.copy()
    dense.X = adata.X.toarray()
    gene_set = ['gene1', 'gene2', 'gene3']
    out_sparse = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, chunk_size=100, reservoir_size=100,
        random_state=0, verbose=False,
    )
    out_dense = get_sccellfie_dataset_threshold(
        dense, gene_set=gene_set, chunk_size=100, reservoir_size=100,
        random_state=0, verbose=False,
    )
    np.testing.assert_allclose(out_sparse.values, out_dense.values, rtol=1e-5, atol=1e-4)


def test_get_sccellfie_dataset_threshold_cell_mask_bool_array():
    adata = create_controlled_adata()
    gene_set = ['gene1', 'gene2', 'gene3']
    mask = np.array([True, True, False, False])

    manual = adata[mask].copy()
    expected = _reference_dataset_threshold(manual.X, list(manual.var_names), gene_set)
    out = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, cell_mask=mask,
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    np.testing.assert_allclose(out['sccellfie_threshold'].values,
                               expected['sccellfie_threshold'].values, rtol=1e-5, atol=1e-4)


def test_get_sccellfie_dataset_threshold_cell_mask_obs_column():
    adata = create_controlled_adata()
    adata.obs['keep'] = [True, True, False, False]
    gene_set = ['gene1', 'gene2', 'gene3']
    out = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, cell_mask='keep',
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    expected = _reference_dataset_threshold(
        adata[adata.obs['keep']].X, list(adata.var_names), gene_set,
    )
    np.testing.assert_allclose(out['sccellfie_threshold'].values,
                               expected['sccellfie_threshold'].values, rtol=1e-5, atol=1e-4)


def test_get_sccellfie_dataset_threshold_use_raw():
    adata = create_controlled_adata()
    gene_set = ['gene1', 'gene2', 'gene3']
    out_raw = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, use_raw=True,
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    out_x = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set,
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    assert_frame_equal(out_raw, out_x)


def test_get_sccellfie_dataset_threshold_already_normalized():
    """If adata.uns['normalization']['method'] == 'total_counts', skip normalization."""
    adata = create_controlled_adata()
    prenorm = adata.copy()
    X = prenorm.X.toarray()
    n_counts = X.sum(axis=1, keepdims=True)
    prenorm.X = sparse.csr_matrix(np.where(n_counts > 0, X / n_counts * 10_000, 0))
    prenorm.uns['normalization'] = {'method': 'total_counts', 'target_sum': 10_000}

    gene_set = ['gene1', 'gene2', 'gene3']
    out = get_sccellfie_dataset_threshold(
        prenorm, gene_set=gene_set,
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    expected = _reference_dataset_threshold(
        prenorm.X, list(prenorm.var_names), gene_set, normalize=False,
    )
    np.testing.assert_allclose(out['sccellfie_threshold'].values,
                               expected['sccellfie_threshold'].values, rtol=1e-5, atol=1e-4)


def test_get_sccellfie_dataset_threshold_target_sum_none_skips_normalization():
    adata = create_controlled_adata()
    gene_set = ['gene1', 'gene2', 'gene3']
    out = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, target_sum=None,
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    expected = _reference_dataset_threshold(
        adata.X, list(adata.var_names), gene_set, normalize=False,
    )
    np.testing.assert_allclose(out['sccellfie_threshold'].values,
                               expected['sccellfie_threshold'].values, rtol=1e-5, atol=1e-4)


def test_get_sccellfie_dataset_threshold_n_counts_key_used():
    """If a precomputed per-cell total is in obs, the function uses it."""
    adata = create_controlled_adata()
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).ravel()
    gene_set = ['gene1', 'gene2', 'gene3']
    out = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, n_counts_key='total_counts',
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    expected = _reference_dataset_threshold(adata.X, list(adata.var_names), gene_set)
    np.testing.assert_allclose(out['sccellfie_threshold'].values,
                               expected['sccellfie_threshold'].values, rtol=1e-5, atol=1e-4)


def test_get_sccellfie_dataset_threshold_correct_genes_applied():
    """CORRECT_GENES['human'] maps e.g. 'MT-CO1' -> 'COX1' before intersecting with gene_set."""
    adata = create_controlled_adata()
    new_names = ['MT-CO1', 'gene2', 'gene3']
    adata.var_names = new_names
    adata.raw = adata.copy()
    gene_set = ['COX1', 'gene2', 'gene3']
    out = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, organism='human',
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    assert 'COX1' in out.index
    assert 'MT-CO1' not in out.index


def test_get_sccellfie_dataset_threshold_gene_set_from_json(tmp_path):
    adata = create_controlled_adata()
    path = tmp_path / 'genes.json'
    with open(path, 'w') as fp:
        json.dump(['gene1', 'gene3'], fp)
    out = get_sccellfie_dataset_threshold(
        adata, gene_set=str(path),
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    assert list(out.index) == ['gene1', 'gene3']


def test_get_sccellfie_dataset_threshold_return_stats():
    adata = create_controlled_adata()
    gene_set = ['gene1', 'gene2', 'gene3']
    out, stats = get_sccellfie_dataset_threshold(
        adata, gene_set=gene_set, return_stats=True,
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    for key in ('percentiles', 'sum_per_gene', 'nnz_per_gene', 'max_per_gene',
                'mean', 'nonzero_mean', 'n_cells', 'n_values_seen', 'reservoir_size_used'):
        assert key in stats
    assert stats['n_cells'] == 4
    assert stats['n_values_seen'] == 11
    assert 25 in stats['percentiles'] and 75 in stats['percentiles']


def test_get_sccellfie_dataset_threshold_no_overlap_raises():
    adata = create_controlled_adata()
    with pytest.raises(ValueError, match='No overlap'):
        get_sccellfie_dataset_threshold(
            adata, gene_set=['not_present_1', 'not_present_2'],
            chunk_size=100, reservoir_size=100, verbose=False,
        )


def test_get_sccellfie_dataset_threshold_compatible_with_compute_gene_scores():
    """The returned DataFrame's first column is positionally consumed by compute_gene_scores."""
    from sccellfie.gene_score import compute_gene_scores
    adata = create_controlled_adata()
    thr = get_sccellfie_dataset_threshold(
        adata, gene_set=['gene1', 'gene2', 'gene3'],
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )
    compute_gene_scores(adata, thresholds=thr)
    assert 'gene_scores' in adata.layers
    assert adata.layers['gene_scores'].shape == adata.X.shape


def test_get_sccellfie_dataset_threshold_custom_percentiles():
    """User-supplied lower/upper percentile bounds drive the clip rule."""
    adata = create_controlled_adata()
    gene_set = ['gene1', 'gene2', 'gene3']
    common = dict(
        adata=adata, gene_set=gene_set,
        chunk_size=100, reservoir_size=100, random_state=0, verbose=False,
    )

    default = get_sccellfie_dataset_threshold(**common)
    explicit_default = get_sccellfie_dataset_threshold(
        **common, lower_percentile=25, upper_percentile=75,
    )
    assert_frame_equal(default, explicit_default)

    wide, wide_stats = get_sccellfie_dataset_threshold(
        **common, lower_percentile=10, upper_percentile=90, return_stats=True,
    )
    assert 10 in wide_stats['percentiles']
    assert 90 in wide_stats['percentiles']
    p10 = wide_stats['percentiles'][10]
    p90 = wide_stats['percentiles'][90]
    # Every threshold produced under the wide rule must lie in [p10, p90]
    # for clipped genes, or equal the raw nonzero_mean for the
    # low-expression escape branch (which still falls within that range
    # for non-zero-max genes; here we just bound by the outer interval).
    thresh = wide['sccellfie_threshold'].values
    nz_mean = wide_stats['nonzero_mean'].values
    in_clip = (thresh >= p10 - 1e-9) & (thresh <= p90 + 1e-9)
    in_escape = np.isclose(thresh, nz_mean)
    assert (in_clip | in_escape).all()
    assert wide.shape == default.shape


def test_get_sccellfie_dataset_threshold_invalid_percentiles_raises():
    adata = create_controlled_adata()
    with pytest.raises(ValueError, match='lower_percentile'):
        get_sccellfie_dataset_threshold(
            adata, gene_set=['gene1', 'gene2', 'gene3'],
            chunk_size=100, reservoir_size=100, verbose=False,
            lower_percentile=80, upper_percentile=20,
        )
    with pytest.raises(ValueError, match='lower_percentile'):
        get_sccellfie_dataset_threshold(
            adata, gene_set=['gene1', 'gene2', 'gene3'],
            chunk_size=100, reservoir_size=100, verbose=False,
            lower_percentile=50, upper_percentile=50,
        )
    with pytest.raises(ValueError, match='lower_percentile'):
        get_sccellfie_dataset_threshold(
            adata, gene_set=['gene1', 'gene2', 'gene3'],
            chunk_size=100, reservoir_size=100, verbose=False,
            lower_percentile=-1, upper_percentile=75,
        )