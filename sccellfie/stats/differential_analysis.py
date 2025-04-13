import scanpy as sc
import numpy as np
import pandas as pd

from collections import defaultdict
from itertools import combinations
from scipy.sparse import issparse
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from sccellfie.preprocessing.adata_utils import get_adata_gene_expression


def cohens_d(group1, group2):
    """
    Calculates Cohen's d effect size for two groups.

    Parameters
    ----------
    group1 : array-like
        Values from the first group of samples.

    group2 : array-like
        Values from the second group of samples.

    Returns
    -------
    d : float
        Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    # Pooled standard deviation
    dof = n1 + n2 - 2
    if dof > 0:
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / dof)
        d = (mean2 - mean1) / pooled_std
    else:
        d = 0
    return d


def _process_de_analysis(adata, condition_key, condition1, condition2, var_names, ct):
    """
    Helper function to process differential expression analysis for a given comparison.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the expression data.

    condition_key : str
        The column name in adata.obs containing the condition information.

    condition1 : str
        The first condition in the comparison.

    condition2 : str
        The second condition in the comparison.

    var_names : list of str
        The list of variable names (e.g. genes) to perform the differential expression
        analysis on.

    ct : str
        The cell type being analyzed.

    Returns
    -------
    results : list of dict
        A list of dictionaries containing the results of the differential expression analysis.
    """
    # Perform Wilcoxon test
    sc.tl.rank_genes_groups(
        adata, groupby=condition_key,
        groups=[condition2], reference=condition1,
        method='wilcoxon', key_added=f"wilcoxon_{ct}"
    )

    # Get results
    df = sc.get.rank_genes_groups_df(adata, group=condition2, key=f"wilcoxon_{ct}")
    df = df[df['names'].isin(var_names)]  # Filter for specified genes

    results = []
    for _, row in df.iterrows():
        gene = row['names']
        group1 = adata[adata.obs[condition_key] == condition1, gene].X
        group2 = adata[adata.obs[condition_key] == condition2, gene].X

        if issparse(group1):
            group1 = group1.toarray().flatten()
        else:
            group1 = group1.flatten()
        if issparse(group2):
            group2 = group2.toarray().flatten()
        else:
            group2 = group2.flatten()

        # Calculate log2 fold change
        mean1, mean2 = np.mean(group1), np.mean(group2)
        fold_change = mean2 / mean1 if mean1 != 0 else 300
        log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0.

        results.append({
            'cell_type': ct,
            'feature': gene,
            'group1': condition1,
            'group2': condition2,
            'log2FC': log2_fold_change,
            'test_statistic': row['scores'],
            'p_value': row['pvals'],
            'cohens_d': cohens_d(group1, group2),
            'n_group1': len(group1),
            'n_group2': len(group2),
            'median_group1': np.median(group1),
            'median_group2': np.median(group2),
            'median_diff': np.median(group2) - np.median(group1)
        })
    return results


def scanpy_differential_analysis(adata, cell_type, cell_type_key, condition_key, condition_pairs=None, var_names=None,
                                 alpha=0.05, min_cells=30, downsample=False, n_iterations=50, agg_method='mean', random_state=None):
    """
    Performs differential expression analysis using Scanpy's rank_genes_groups function.

    Parameters
    ----------
    adata : AnnData object
        Annotated data matrix containing the expression data.

    cell_type : str or None
        The cell type to analyze. If None, analysis is performed for all cell types.

    cell_type_key : str
        The column name in adata.obs containing the cell type information.

    condition_key : str
        The column name in adata.obs containing the condition information.

    condition_pairs : list of tuples, optional (default: None)
        The pairs of conditions to compare. If None, all pairs of conditions are compared.

    var_names : list of str, optional (default: None)
        The list of variable names (e.g. genes) to perform the differential expression
        analysis on. If None, all genes are used.

    alpha : float, optional (default: 0.05)
        The significance level for the multiple testing correction.

    min_cells : int, optional (default: 30)
        Minimum number of cells required in each group for comparison.

    downsample : bool, optional (default: False)
        Whether to perform downsampling to balance group sizes.

    n_iterations : int, optional (default: 50)
        Number of subsampling iterations if downsample=True.

    agg_method : str, optional (default: 'mean')
        Method to aggregate results across iterations ('mean' or 'median').

    random_state : int, optional (default: None)
        Random seed for reproducibility of downsampling.

    Returns
    -------
    df_results : pandas.DataFrame
        A DataFrame containing the results of the differential expression analysis with columns:
            - cell_type: The analyzed cell type
            - feature: Name of the analyzed feature
            - group1: First condition in the comparison
            - group2: Second condition in the comparison
            - log2FC: Log2 fold change between means of conditions
            - test_statistic: Wilcoxon test statistic
            - p_value: Raw p-value
            - adj_p_value: BH-corrected p-value
            - cohens_d: Effect size (Cohen's d)
            - n_group1: Number of observations in group1
            - n_group2: Number of observations in group2
            - median_group1: Median expression in group1
            - median_group2: Median expression in group2
            - median_diff: Difference in medians (group2 - group1)
    """
    excluded_cells = defaultdict(list)

    if cell_type is None:
        cell_types = adata.obs[cell_type_key].unique()
    else:
        cell_types = [cell_type]

    all_results = []
    total_combinations = len(cell_types) * (
        len(condition_pairs) if condition_pairs else len(list(combinations(adata.obs[condition_key].unique(), 2)))
    )

    with tqdm(total=total_combinations, desc="Processing DE analysis") as pbar:
        for ct in cell_types:
            # Filter for the specific cell type
            adata_subset = adata[adata.obs[cell_type_key] == ct].copy()

            if var_names is None:
                var_names = adata_subset.var_names.tolist()

            # If condition_pairs is None, compare all pairs of conditions
            if condition_pairs is None:
                unique_conditions = adata_subset.obs[condition_key].unique()
                condition_pairs = list(combinations(unique_conditions, 2))

            for condition1, condition2 in condition_pairs:
                n1 = np.sum(adata_subset.obs[condition_key] == condition1)
                n2 = np.sum(adata_subset.obs[condition_key] == condition2)

                if min(n1, n2) < min_cells:
                    excluded_cells[ct].append(f"{condition1} vs {condition2} (n1={n1}, n2={n2})")
                    pbar.update(1)
                    continue

                if downsample:
                    n_small = min(n1, n2)
                    iter_results = []

                    for i in range(n_iterations):
                        rng = np.random.RandomState(
                            random_state + i) if random_state is not None else np.random.RandomState()
                        idx1 = rng.choice(np.where(adata_subset.obs[condition_key] == condition1)[0], n_small)
                        idx2 = rng.choice(np.where(adata_subset.obs[condition_key] == condition2)[0], n_small)

                        temp_adata = adata_subset[np.concatenate([idx1, idx2])]  # .copy()
                        results = _process_de_analysis(temp_adata, condition_key, condition1, condition2, var_names, ct)

                        for r in results:
                            r['iteration'] = i
                        iter_results.extend(results)

                    df_iter = pd.DataFrame(iter_results)
                    agg_results = df_iter.groupby(['cell_type', 'feature', 'group1', 'group2']).agg({
                        'log2FC': agg_method,
                        'test_statistic': agg_method,
                        'p_value': agg_method,
                        'cohens_d': agg_method,
                        'n_group1': 'first',  # These values will be the same across iterations
                        'n_group2': 'first',
                        'median_group1': agg_method,
                        'median_group2': agg_method,
                        'median_diff': agg_method
                    }).reset_index()
                    all_results.extend(agg_results.to_dict('records'))

                else:
                    results = _process_de_analysis(adata_subset, condition_key, condition1, condition2, var_names, ct)
                    all_results.extend(results)

                pbar.update(1)

    if excluded_cells:
        print("\nExcluded comparisons due to insufficient cells:")
        for ct, comparisons in excluded_cells.items():
            print(f"\n{ct}:")
            for comp in comparisons:
                print(f"  - {comp}")

    df_results = pd.DataFrame(all_results)
    if not df_results.empty:
        df_results['adj_p_value'] = multipletests(df_results['p_value'], method='fdr_bh', alpha=alpha)[1]

    return df_results


def pairwise_differential_analysis(adata, groupby, var_names=None, order=None, alternative='two-sided', alpha=0.05):
    """
    Performs pairwise Wilcoxon tests for each feature between all group pairs.
    This functions does not perform the test in a cell type-wise manner.
    For that, use ´scanpy_differential_analysis´.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the expression data.

    groupby : str
        Column in adata.obs containing group labels.

    var_names : list, optional (default: None)
        List of feature names to test. If None, all features are tested.

    order : list, optional (default: None)
        Specific order of groups to test. If None, groups are sorted.

    alternative : str, optional (default: 'two-sided')
        Alternative hypothesis for the Wilcoxon rank-sum test.
        Options are 'two-sided', 'greater', 'less'.

    alpha : float, optional (default: 0.05)
        Significance level for multiple testing correction.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the results with the same columns as scanpy_differential_analysis
        (except 'cell_type') for consistency:
            - feature: Name of the analyzed feature
            - group1: First condition in the comparison
            - group2: Second condition in the comparison
            - log2FC: Log2 fold change between conditions
            - test_statistic: Wilcoxon test statistic
            - p_value: Raw p-value
            - adj_p_value: BH-corrected p-value
            - cohens_d: Effect size (Cohen's d)
            - n_group1: Number of observations in group1
            - n_group2: Number of observations in group2
            - median_group1: Median expression in group1
            - median_group2: Median expression in group2
            - median_diff: Difference in medians (group2 - group1)
    """
    # Lists to store results for DataFrame
    results_list = []

    # Get groups
    groups = adata.obs[groupby]
    group_order = order if order else sorted(groups.unique())

    # Get feature names
    if var_names is None:
        var_names = adata.var_names.tolist()

    # Process each feature
    for var_name in tqdm(var_names, desc="Processing features"):
        # Get expression values
        expression = get_adata_gene_expression(adata, var_name)

        # Filter for ordered groups if specified
        if order is not None:
            mask = groups.isin(order)
            expr_filtered = expression[mask]
            groups_filtered = groups[mask]
        else:
            expr_filtered = expression
            groups_filtered = groups

        # Perform pairwise comparisons
        for i, group1 in enumerate(group_order):
            for group2 in group_order[i + 1:]:
                values1 = expr_filtered[groups_filtered == group1]
                values2 = expr_filtered[groups_filtered == group2]

                # Remove NaN values from NumPy arrays
                values1 = values1[~np.isnan(values1)]
                values2 = values2[~np.isnan(values2)]

                # Perform Wilcoxon rank-sum test
                statistic, pvalue = ranksums(values2, values1, alternative=alternative)

                # Calculate log2 fold change
                mean1, mean2 = np.mean(values1), np.mean(values2)
                fold_change = mean2 / mean1 if mean1 != 0 else 300
                log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0.

                # Store results
                results_list.append({
                    'feature': var_name,
                    'group1': group1,
                    'group2': group2,
                    'log2FC': log2_fold_change,
                    'test_statistic': statistic,
                    'p_value': pvalue,
                    'cohens_d': cohens_d(values1, values2),
                    'n_group1': len(values1),
                    'n_group2': len(values2),
                    'median_group1': np.median(values1),
                    'median_group2': np.median(values2),
                    'median_diff': np.median(values2) - np.median(values1)
                })

    # Create DataFrame
    df = pd.DataFrame(results_list)

    # Perform global multiple testing correction
    df['adj_p_value'] = multipletests(df['p_value'], alpha=alpha, method='fdr_bh')[1]

    return df