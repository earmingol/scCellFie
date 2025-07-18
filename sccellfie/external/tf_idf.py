import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.stats import hypergeom
from scipy.sparse import issparse, csr_matrix
from statsmodels.stats.multitest import multipletests


def quick_markers(adata, cluster_key, cell_groups=None, n_markers=10, fdr=0.01, express_cut=0.9, r_output=False):
    """
    Identifies top N markers for each cluster in an AnnData object using a TF-IDF-based strategy.
    Implemented as in the SoupX library for R.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix from Scanpy.

    cluster_key : str
        Key in adata.obs for the cluster labels.

    cell_groups : list, optional (default: None)
        List of cell groups to be compared in the analysis.

    n_markers : int, optional (default: 10)
        Number of marker genes to return per cluster.

    fdr : float, optional (default: 0.01)
        False discovery rate for the hypergeometric test.

    express_cut : float, optional (default: 0.9)
        Value above which a gene is considered expressed.

    r_output : bool, optional (default: False)
        Whether reporting the same exact column names as the SoupX version.

    Returns
    -------
    markers : pandas.DataFrame
        A pandas.DataFrame with top N markers for each cluster and their statistics.
    """
    if cell_groups is not None:
        adata_ = adata[adata.obs[cluster_key].isin(cell_groups)]
    else:
        adata_ = adata

    # Convert to CSR matrix if necessary and binarize the expression data
    toc = csr_matrix(adata_.X) if not issparse(adata_.X) else adata_.X
    toc_bin = (toc > express_cut).astype(int)

    # Cluster information
    clusters = pd.Categorical(adata_.obs[cluster_key]).codes
    unique_clusters = np.unique(clusters)
    cl_counts = np.asarray([np.sum(clusters == cl) for cl in unique_clusters]).reshape(-1 ,1)

    # Calculate observed and total frequency
    n_obs = np.asarray([np.asarray(toc_bin[clusters == cl, :].sum(axis=0)).flatten() for cl in unique_clusters])
    n_tot = n_obs.sum(axis=0)

    # Term Frequency (TF), Inverse Document Frequency (IDF) and TF-IDF
    tf = n_obs / cl_counts
    idf = np.log(len(clusters) / n_tot)
    tf_idf = tf * idf

    # Calculate additional metrics
    gene_freq_outside_cluster = (n_tot - n_obs) / (len(clusters) - cl_counts)
    gene_freq_global = n_tot / len(clusters)

    # Calculate second-best TF score and corresponding cluster name
    second_best_tf = np.zeros_like(tf)
    second_best_cluster_idx = np.zeros(tf.shape[1], dtype=int)
    for gene_idx in range(tf.shape[1]):
        tf_scores = tf[:, gene_idx]
        second_best_idx = np.argsort(tf_scores)[-2]  # Get index of second-highest value
        second_best_tf[:, gene_idx] = tf_scores[second_best_idx]
        second_best_cluster_idx[gene_idx] = unique_clusters[second_best_idx]

    # P-values
    p_values = np.array \
        ([hypergeom.sf(n_obs[i] - 1, len(clusters), n_tot, cl_counts[i]) for i in range(len(unique_clusters))])

    # FDR correction using statsmodels (global across all gene-cluster pairs)
    p_flat = p_values.flatten()
    reject, q_flat, _, _ = multipletests(p_flat, alpha=fdr, method='fdr_bh')
    q_values = q_flat.reshape(p_values.shape)

    # Select top N markers by iterating over columns of p-values matrix
    top_markers = {cl: [] for cl in unique_clusters}
    for gene_idx in range(tf_idf.shape[1]):
        for cl in unique_clusters:
            # Filter genes by FDR (statsmodels handles NaN automatically)
            q_val = q_values[cl, gene_idx]
            if not np.isnan(q_val) and q_val < fdr:
                top_markers[cl].append((gene_idx, tf_idf[cl, gene_idx]))

    # Sort and select top genes for each cluster
    for cl in top_markers:
        top_markers[cl].sort(key=lambda x: x[1], reverse=True)  # Sort by TF-IDF
        top_markers[cl] = [gene_idx for gene_idx, _ in top_markers[cl][:n_markers]]

    # Constructing the output DataFrame
    marker_data = []
    for cl, markers in top_markers.items():
        for gene_idx in markers:
            gene = adata.var_names[gene_idx]
            second_best_cl = second_best_cluster_idx[gene_idx]
            marker_data.append({
                'gene': gene,
                'cluster': adata.obs[cluster_key].cat.categories[cl],
                'tf': tf[cl, gene_idx],
                'idf': idf[gene_idx],
                'tf_idf': tf_idf[cl, gene_idx],
                'gene_frequency_outside_cluster': gene_freq_outside_cluster[cl, gene_idx],
                'gene_frequency_global': gene_freq_global[gene_idx],
                'second_best_tf': second_best_tf[cl, gene_idx],
                'second_best_cluster': adata.obs[cluster_key].cat.categories[second_best_cl],
                'pval': p_values[cl, gene_idx],
                'qval': q_values[cl, gene_idx]
            })

    markers = pd.DataFrame(marker_data)
    if markers.shape == (0,0):
        markers = pd.DataFrame(columns=['gene', 'cluster', 'tf', 'idf', 'tf_idf', 'gene_frequency_outside_cluster',
                                        'gene_frequency_global', 'second_best_tf', 'second_best_cluster', 'pval', 'qval'])

    if r_output:
        cols = ['gene', 'cluster', 'tf', 'gene_frequency_outside_cluster', 'second_best_tf', 'gene_frequency_global', 'second_best_cluster', 'tf_idf', 'idf', 'qval']
        markers = markers[cols]
        markers.columns = ['gene', 'cluster', 'geneFrequency', 'geneFrequencyOutsideCluster',
                           'geneFrequencySecondBest', 'geneFrequencyGlobal', 'secondBestClusterName', 'tfidf', 'idf', 'qval']
    return markers


def filter_tfidf_markers(df, tf_col='tf', idf_col='idf', tfidf_threshold=None, tfidf_col='tf_idf',
                         tf_ratio=None, second_best_tf_col='second_best_tf', group_col='cluster', second_best_group_col='second_best_cluster'):
    """
    Filters the top N markers for each cluster based on a hyperbolic curve fit to the TF-IDF values.
    Additional filtering can be applied based on the TF-IDF threshold and the ratio of the
    TF score to the second-best TF score.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the marker data. See `sccellfie.preprocessing.quick_markers` for details.

    tf_col : str, optional (default: 'tf')
        Column name for the Term Frequency (TF) values.

    idf_col : str, optional (default: 'idf')
        Column name for the Inverse Document Frequency (IDF) values.

    tfidf_threshold : float, optional (default: None)
        Threshold for the TF-IDF values. If provided, only markers with TF-IDF values above this threshold are kept.
        A value of 0.3 is recommended for most datasets.

    tfidf_col : str, optional (default: 'tf_idf')
        Column name for the TF-IDF values. Used for filtering based on the TF-IDF threshold.

    tf_ratio : float, optional (default: None)
        Threshold for the ratio of the TF score to the second-best TF score. If provided, only markers with a ratio
        above this threshold are kept. A value of 1.2 is recommended for most datasets.

    second_best_tf_col : str, optional (default: 'second_best_tf')
        Column name for the second-best TF values. Used for filtering based on the TF ratio.

    group_col : str, optional (default: 'cluster')
        Column name for the cluster labels. Used for filtering based on the TF ratio. This
        is to keep markers when the cluster equals the second-best cluster (very specific marker).

    second_best_group_col : str, optional (default: 'second_best_cluster')
        Column name for the second-best cluster labels. Used for filtering based on the TF ratio.
        This is to keep markers when the cluster equals the second-best cluster (very specific marker).

    Returns
    -------
    filtered_df : pandas.DataFrame
        DataFrame containing the filtered markers.

    theoretical_curve : tuple
        Tuple containing the x and y values of the theoretical hyperbolic curve.
    """
    # Define hyperbola function
    def hyperbola(x, a, b, c):
        return c / (x + a) + b

    # Fit hyperbola to the data
    x = df[tf_col].values
    y = df[idf_col].values
    popt, _ = curve_fit(hyperbola, x, y, p0=[0.1, 0, 0.5], bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))

    # Calculate theoretical values
    lin_x = np.linspace(df[tf_col].min(), df[tf_col].max(), df.shape[0])
    lin_y = hyperbola(lin_x, *popt)
    theoretical_curve = (lin_x, lin_y)

    # Select points above the curve
    exp_y = hyperbola(df[tf_col], *popt)
    above_curve_mask = df[idf_col] >= exp_y
    if tfidf_threshold is not None:
        above_curve_mask = above_curve_mask & (df[tfidf_col] > tfidf_threshold)

    if tf_ratio is not None:
        above_curve_mask = above_curve_mask & ((df[tf_col] / df[second_best_tf_col] > tf_ratio) | (df[group_col] == df[second_best_group_col]))

    return df[above_curve_mask], theoretical_curve