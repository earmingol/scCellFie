import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def create_volcano_plot(de_results, effect_threshold=0.75, padj_threshold=0.05, contrast=None, effect_col='cohens_d',
                        effect_title="Cohen's d", save=None):
    """
    Creates a volcano plot for differential analysis results.

    Parameters
    ----------
    de_results : DataFrame
        A DataFrame containing the results of the differential analysis.
        The DataFrame should have the following columns: 'feature', 'contrast', 'log2FC', 'test_statistic',
        'p_value', 'cohens_d', and 'adj_p_value'.

    effect_threshold : float, optional (default: 0.75)
        The threshold for the effect size (e.g., log2 fold change or Cohen's d) to consider a variable significant.

    padj_threshold : float, optional (default: 0.05)
        The threshold for the adjusted p-value to consider a variable significant.

    contrast : str, optional (default: None)
        The specific contrast to plot. If None, all contrasts are plotted.
        By contrast, we mean the comparison between two conditions (e.g., 'A vs B').

    effect_col : str, optional (default: 'cohens_d')
        The column in de_results that contains the effect size values.

    effect_title : str, optional (default: "Cohen's d")
        The title to use for the effect size in the plot.

    save : str, optional (default: None)
        The file path to save the plot. If None, the plot is not saved.
        A file extension (e.g., '.png') can be provided to specify the file format.

    Returns
    -------
    significant_points : list
        A list of indices corresponding to the significant points in the volcano plot.

    Notes
    -----
    This function creates a volcano plot to visualize the results of a differential analysis.
    The x-axis represents the effect size (e.g., log2 fold change or Cohen's d), and the y-axis
    represents the -log10(adjusted p-value). Significant points are highlighted in red.
    """
    # Filter for specific contrast if provided
    if contrast:
        df = de_results[de_results['contrast'] == contrast].copy()
    else:
        df = de_results.copy()

    # Calculate -log10(padj)
    df['neg_log_padj'] = -np.log10(df['adj_p_value'])
    df['neg_log_padj'] = df['neg_log_padj'].replace(np.inf, 300)

    # Create the volcano plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define significance filter
    sig_filter = (df['adj_p_value'] < padj_threshold) & (df[effect_col].abs() >= effect_threshold)

    # Plot non-significant points
    ax.scatter(df[~sig_filter][effect_col], df[~sig_filter]['neg_log_padj'],
               color='gray', alpha=0.5, label='Not significant')

    # Plot significant points
    significant_points = df[sig_filter]
    ax.scatter(significant_points[effect_col], significant_points['neg_log_padj'],
               color='red', alpha=0.8, label='Significant')

    # Add labels and title
    ax.set_xlabel(effect_title)
    ax.set_ylabel('-log10(adj. p-value)')
    title = 'Volcano Plot' if contrast is None else f'{contrast}'
    ax.set_title(title)

    # Set the x-axis limits to center the plot around zero
    max_lfc = np.ceil(np.nanmax(df[effect_col].abs().values))
    # print(max_lfc)
    ax.set_xlim(-max_lfc, max_lfc)

    # Add threshold lines
    ax.axhline(-np.log10(padj_threshold), color='darkgray', linestyle='--')
    ax.axvline(-effect_threshold, color='darkgray', linestyle='--')
    ax.axvline(effect_threshold, color='darkgray', linestyle='--')

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300,
                    bbox_inches='tight')

    return significant_points.sort_values(effect_col, ascending=True).index.tolist()


def create_comparative_violin(adata, significant_features, group1, group2, condition_key,
                              celltype, cell_type_key, xlabel='Feature', ylabel='Metabolic Activity',
                              title=None, figsize=(16, 7), fontsize=10,
                              palette=['coral', 'lightsteelblue'], filename=None, dpi=300):
    """
    Compares features between two groups for a specific cell type in an AnnData object and creates a violin plot.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing the data.

    significant_features : list
        List of significant features from the volcano plot function. Each item is a tuple where
        the feature name is in the second position (index 1).

    group1 : str
        The name of the first group to compare.

    group2 : str
        The name of the second group to compare.

    condition_key : str
        The column name in adata.obs containing the condition information.

    celltype : str
        The cell type to analyze.

    cell_type_key : str
        The column name in adata.obs containing the cell type information.

    xlabel : str, optional (default: 'Feature')
        The label for the x-axis.

    ylabel : str, optional (default: 'Metabolic Activity')
        The label for the y-axis.

    title : str, optional (default: None)
        The title for the plot. If None, a default title is used.

    figsize : tuple, optional (default: (16, 7))
        The figure size.

    fontsize : int, optional (default: 10)
        The font size for the labels and legend.

    palette : list, optional (default: ['coral', 'lightsteelblue'])
        The color palette for the plot. Each color corresponds to a group or condition.

    filename : str, optional (default: None)
        The file path to save the plot. If None, the plot is not saved.

    dpi : int, optional (default: 300)
        The resolution of the saved figure.

    Returns
    -------
    fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        The matplotlib Figure and Axes objects for the plot.

    feature_df : pandas.DataFrame
        A DataFrame containing the data used for the plot. Each row corresponds to a feature and group.
    """
    # Filter cells for each group
    cells_group1 = adata.obs.loc[(adata.obs[condition_key] == group1) & (adata.obs[cell_type_key] == celltype)].index
    cells_group2 = adata.obs.loc[(adata.obs[condition_key] == group2) & (adata.obs[cell_type_key] == celltype)].index

    # Extract feature names from the significant_features list
    feature_names = [feature[1] for feature in significant_features]

    # Compile data for significant features
    feature_dfs = []
    for feature in feature_names:
        series1 = adata[cells_group1].to_df()[feature]
        series2 = adata[cells_group2].to_df()[feature]
        df = pd.concat([series1, series2], axis=0)
        df = df.reset_index(level=0)
        df = df.rename(columns={feature: ylabel})
        df['Feature'] = feature
        df['Group'] = pd.Series([group1] * len(series1) + [group2] * len(series2))
        feature_dfs.append(df)

    feature_df = pd.concat(feature_dfs, axis=0)

    # Create the violin plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=feature_df, x='Feature', y=ylabel, hue='Group',
                   split=True, inner="quart", linewidth=0.5, density_norm="width",
                   palette=palette, order=feature_names, ax=ax)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90, rotation_mode='anchor', ha='right', va='center',
                       fontsize=fontsize - 4)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
    plt.legend(frameon=False, fontsize=fontsize - 2)

    if title is None:
        title = f"Feature Comparison: {group1} vs {group2} ({celltype})"
    plt.title(title, fontsize=fontsize + 2)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')

    return fig, ax