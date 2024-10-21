import numpy as np
import pandas as pd


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