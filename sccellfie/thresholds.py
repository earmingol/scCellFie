import numpy as np
import pandas as pd


def get_local_percentile_threshold(adata, percentile=0.75, lower_bound=0.25, upper_bound=None, exclude_zeros=False):
    if exclude_zeros:
        X = np.ma.masked_equal(adata.X.toarray(), 0)
        thresholds = np.quantile(X, q=percentile, axis=0).data
    else:
        X = adata.X.toarray()
        thresholds = np.quantile(X, q=percentile, axis=0)
    if isinstance(percentile, list):
        columns = ['threshold-{}'.format(p) for p in percentile]
    else:
        columns = ['threshold-{}'.format(percentile)]
    thresholds = pd.DataFrame(thresholds.T, index=adata.var_names, columns=columns)

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
            thresholds[thresholds < ub] = ub[thresholds < ub]
        else:
            thresholds[thresholds < upper_bound] = upper_bound
    return thresholds


def get_global_percentile_threshold(adata, percentile=0.75, lower_bound=0.25, upper_bound=None, exclude_zeros=False):
    if exclude_zeros:
        X = np.ma.masked_equal(adata.X.toarray(), 0)
        thresholds = np.quantile(X, q=percentile)
    else:
        X = adata.X.toarray()
        thresholds = np.quantile(X, q=percentile)
    if isinstance(percentile, list):
        columns = ['threshold-{}'.format(p) for p in percentile]
    else:
        columns = ['threshold-{}'.format(percentile)]
    thresholds = pd.DataFrame(thresholds.T, index=adata.var_names, columns=columns)

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
            thresholds[thresholds < ub] = ub[thresholds < ub]
        else:
            thresholds[thresholds < upper_bound] = upper_bound
    return thresholds


def get_local_mean_threshold(adata, lower_bound=0.25, upper_bound=None, exclude_zeros=False):
    if exclude_zeros:
        X = np.ma.masked_equal(adata.X.toarray(), 0)
        thresholds = np.ma.mean(X, axis=0).data
    else:
        X = adata.X.toarray()
        thresholds = np.nanmean(X, axis=0)
    columns = ['threshold-mean']
    thresholds = pd.DataFrame(thresholds.T, index=adata.var_names, columns=columns)

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
            thresholds[thresholds < ub] = ub[thresholds < ub]
        else:
            thresholds[thresholds < upper_bound] = upper_bound
    return thresholds


def get_global_mean_threshold(adata, lower_bound=0.25, upper_bound=None, exclude_zeros=False):
    if exclude_zeros:
        X = np.ma.masked_equal(adata.X.toarray(), 0)
        thresholds = np.ma.mean(X)
    else:
        X = adata.X.toarray()
        thresholds = np.nanmean(X)
    columns = ['threshold-mean']
    thresholds = pd.DataFrame(thresholds.T, index=adata.var_names, columns=columns)

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
            thresholds[thresholds < ub] = ub[thresholds < ub]
        else:
            thresholds[thresholds < upper_bound] = upper_bound
    return thresholds


def set_manual_threshold(adata, threshold):
    if isinstance(threshold, list):
        assert len(adata.var_names) == len(threshold), "The len of threshold must be the same as gene number in adata"
        thresholds = pd.DataFrame(np.asarray(threshold).reshape(-1, 1), index=adata.var_names, columns=['threshold-manual'])
    else:
        thresholds = pd.DataFrame(data={'threshold-manual': [threshold]}, index=adata.var_names)
    return thresholds