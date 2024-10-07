import scanpy as sc
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import combinations
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests


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
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def scanpy_differential_analysis(adata, cell_type, cell_type_key, condition_key, condition_pairs=None, var_names=None,
                                 alpha=0.05):
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

    Returns
    -------
    df_results : DataFrame
        A DataFrame containing the results of the differential expression analysis.
        The DataFrame has the following columns: 'cell_type', 'feature', 'contrast', 'log2FC',
        'test_statistic', 'p_value', 'cohens_d', and 'adj_p_value'.
    """
    if cell_type is None:
        cell_types = adata.obs[cell_type_key].unique()
    else:
        cell_types = [cell_type]

    all_results = []

    for ct in cell_types:
        # Filter for the specific cell type
        adata_subset = adata[adata.obs[cell_type_key] == ct].copy()

        if var_names is None:
            var_names = adata_subset.var_names.tolist()

        # If condition_pairs is None, compare all pairs of conditions
        if condition_pairs is None:
            unique_conditions = adata_subset.obs[condition_key].unique()
            condition_pairs = list(combinations(unique_conditions, 2))

        results = []

        for condition1, condition2 in condition_pairs:
            print(f"Processing contrast: {condition1} vs {condition2} for cell type: {ct}")

            # Perform Wilcoxon test
            sc.tl.rank_genes_groups(adata_subset, groupby=condition_key,
                                    groups=[condition2], reference=condition1,
                                    method='wilcoxon', key_added=f"wilcoxon_{ct}")

            # Get results
            df = sc.get.rank_genes_groups_df(adata_subset, group=condition2, key=f"wilcoxon_{ct}")
            df = df[df['names'].isin(var_names)]  # Filter for specified genes

            for _, row in df.iterrows():
                gene = row['names']
                group1 = adata_subset[adata_subset.obs[condition_key] == condition1, gene].X
                group2 = adata_subset[adata_subset.obs[condition_key] == condition2, gene].X

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

                # Calculate Cohen's d
                effect_size = cohens_d(group2, group1)

                results.append({
                    'cell_type': ct,
                    'feature': gene,
                    'contrast': f"{condition1} vs {condition2}",
                    'log2FC': log2_fold_change,
                    'test_statistic': row['scores'],
                    'p_value': row['pvals'],
                    'cohens_d': effect_size
                })

        all_results.extend(results)

    # Convert to DataFrame and apply BH correction
    df_results = pd.DataFrame(all_results)
    if not df_results.empty:
        df_results['adj_p_value'] = multipletests(df_results['p_value'], method='fdr_bh', alpha=alpha)[1]

    return df_results.set_index(['cell_type', 'feature'])