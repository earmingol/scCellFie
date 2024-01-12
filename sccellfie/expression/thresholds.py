import numpy as np
import pandas as pd


def get_local_percentile_threshold(adata, percentile=0.75, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()
    if exclude_zeros:
        X = np.ma.masked_equal(X, 0)
        thresholds = np.quantile(X, q=percentile, axis=0, method='midpoint').data
    else:
        thresholds = np.quantile(X, q=percentile, axis=0, method='midpoint')
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
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()
    if exclude_zeros:
        X = np.ma.masked_equal(X, 0)
        thresholds = np.quantile(X, q=percentile, method='midpoint')
    else:
        thresholds = np.quantile(X, q=percentile, method='midpoint')
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


def get_local_mean_threshold(adata, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()
    if exclude_zeros:
        X = np.ma.masked_equal(X, 0)
        thresholds = np.ma.mean(X, axis=0).data
    else:
        thresholds = np.nanmean(X, axis=0)
    columns = ['threshold-mean']
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


def get_global_mean_threshold(adata, lower_bound=1e-5, upper_bound=None, exclude_zeros=False, use_raw=False):
    if use_raw:
        X = adata.raw.X.toarray()
    else:
        X = adata.X.toarray()
    if exclude_zeros:
        X = np.ma.masked_equal(X, 0)
        thresholds = np.ma.mean(X)
    else:
        thresholds = np.nanmean(X)
    columns = ['threshold-mean']
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


def set_manual_threshold(adata, threshold):
    if isinstance(threshold, list):
        assert len(adata.var_names) == len(threshold), "The len of threshold must be the same as gene number in adata"
        thresholds = pd.DataFrame(np.asarray(threshold).reshape(-1, 1), index=adata.var_names, columns=['threshold-manual'])
    else:
        thresholds = pd.DataFrame(data={'threshold-manual': [threshold]}, index=adata.var_names, dtype=float)
    return thresholds