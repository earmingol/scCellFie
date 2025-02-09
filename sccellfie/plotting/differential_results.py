import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from matplotlib.lines import Line2D


def create_volcano_plot(de_results, effect_threshold=0.75, padj_threshold=0.05, cell_type=None, group1=None,
                        group2=None, effect_col='cohens_d', effect_title="Cohen's d", wrapped_title_length=50,
                        save=None, dpi=300, tight_layout=True):
    """
    Creates a volcano plot for differential analysis results.

    Parameters
    ----------
    de_results : pd.DataFrame
        A DataFrame containing the results of the differential analysis.
        Required columns: 'feature', 'adj_p_value', and the column specified in effect_col.
        Optional columns: 'cell_type', 'group1', 'group2'.

    effect_threshold : float, optional (default: 0.75)
        The threshold for the effect size (e.g., log2 fold change or Cohen's d)
        to consider a variable significant.

    padj_threshold : float, optional (default: 0.05)
        The threshold for the adjusted p-value to consider a variable significant.

    cell_type : str, optional (default: None)
        The specific cell type to plot. If None and cell_type column exists,
        all cell types are plotted.

    group1 : str, optional (default: None)
        The first group in the comparison. If None, all group1 values are included.

    group2 : str, optional (default: None)
        The second group in the comparison. If None, all group2 values are included.

    effect_col : str, optional (default: 'cohens_d')
        The column in de_results that contains the effect size values.

    effect_title : str, optional (default: "Cohen's d")
        The title to use for the effect size in the plot.

    wrapped_title_length : int, optional (default: 50)
        The maximum number of characters per line in the title.

    save : str, optional (default: None)
        The file path to save the plot. If None, the plot is not saved.
        A file extension (e.g., '.png') can be provided to specify the file format.

    dpi : int, optional (default: 300)
        The resolution of the saved figure.

    tight_layout : bool, optional (default: True)
        Whether to use tight layout for the plot.

    Returns
    -------
    list
        A list of feature names that are considered significant based on the
        provided thresholds, sorted by effect size in ascending order.
        Returns an empty list if no significant features are found.

    Notes
    -----
    This function creates a volcano plot where:
    - x-axis represents the effect size (e.g., log2 fold change or Cohen's d)
    - y-axis represents the -log10(adjusted p-value)
    - Gray points indicate non-significant features
    - Red points indicate significant features that pass both thresholds
    - Dashed lines indicate the significance thresholds
    """
    # Create a copy of the results to avoid modifying the original
    df = de_results.copy()

    # Filter based on cell type if provided and if column exists
    if cell_type is not None and 'cell_type' in df.columns:
        df = df[df['cell_type'] == cell_type]

    # Filter based on groups if provided
    if group1 is not None:
        df = df[df['group1'] == group1]
    if group2 is not None:
        df = df[df['group2'] == group2]

    # Calculate -log10(padj) for y-axis
    df['neg_log_padj'] = -np.log10(df['adj_p_value'])
    # Replace infinite values with a large number for visualization
    df['neg_log_padj'] = df['neg_log_padj'].replace(np.inf, 300)

    # Create the volcano plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define significance filter based on thresholds
    sig_filter = (df['adj_p_value'] < padj_threshold) & \
                 (df[effect_col].abs() >= effect_threshold)

    # Plot non-significant points in gray
    ax.scatter(df[~sig_filter][effect_col], df[~sig_filter]['neg_log_padj'],
               color='gray', alpha=0.5, label='Not significant')

    # Get and plot significant points in red
    significant_points = df[sig_filter].copy()  # Make a copy to avoid SettingWithCopyWarning
    if not significant_points.empty:
        ax.scatter(significant_points[effect_col], significant_points['neg_log_padj'],
                   color='red', alpha=0.8, label='Significant')

    # Add labels and title
    ax.set_xlabel(effect_title)
    ax.set_ylabel('-log10(adj. p-value)')

    # Create title based on filters
    title_parts = []
    if cell_type is not None:
        title_parts.append(f'Cell Type: {cell_type}')
    if group1 is not None and group2 is not None:
        title_parts.append(f'{group1} vs {group2}')

    # Set and wrap the title
    title = 'Volcano Plot' if not title_parts else ' - '.join(title_parts)
    wrapped_title = "\n".join(textwrap.wrap(title, width=wrapped_title_length))
    ax.set_title(wrapped_title)

    # Set the x-axis limits to center the plot around zero
    max_lfc = np.ceil(np.nanmax(df[effect_col].abs().values))
    ax.set_xlim(-max_lfc, max_lfc)

    # Add threshold lines
    ax.axhline(-np.log10(padj_threshold), color='darkgray', linestyle='--')
    ax.axvline(-effect_threshold, color='darkgray', linestyle='--')
    ax.axvline(effect_threshold, color='darkgray', linestyle='--')

    # Add legend and grid
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    # Apply tight layout if requested
    if tight_layout:
        plt.tight_layout()

    # Save the plot if a save path is provided
    if save:
        from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir

        # Get directory and base filename
        dir, basename = _get_file_dir(save)
        os.makedirs(dir, exist_ok=True)
        format = _get_file_format(save)

        # Update save name to include filters
        save_parts = []
        if cell_type:
            save_parts.append(cell_type)
        if group1 and group2:
            save_parts.append(f'{group1}_vs_{group2}')
        suffix = '_'.join(save_parts)
        suffix = f'_{suffix}' if suffix else ''

        # Construct the final filename with 'volcano_' prefix
        final_filename = f'volcano_{basename}{suffix}.{format}'
        final_path = os.path.join(dir, final_filename)

        # Save the figure
        plt.savefig(final_path, dpi=dpi, bbox_inches='tight')

    # Return sorted list of significant features
    return (significant_points.sort_values(effect_col, ascending=True)
            .feature.unique().tolist() if not significant_points.empty else [])


def create_comparative_violin(adata, significant_features, group1, group2, condition_key,
                              celltype, cell_type_key, xlabel='Feature', ylabel='Metabolic Activity',
                              title=None, wrapped_title_length=50, figsize=(16, 7), fontsize=10, violin_cut=0,
                              palette=['coral', 'lightsteelblue'], lgd_bbox_to_anchor=(1.05, 1), lgd_loc='upper left',
                              save=None, dpi=300, tight_layout=True):
    """
    Compares features between two groups for a specific cell type in an AnnData object and creates a violin plot.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing the data.

    significant_features : list
        List of significant feature names from the volcano plot function, sorted by effect size.

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

    wrapped_title_length : int, optional (default: 50)
        The maximum number of characters per line in the title.

    figsize : tuple, optional (default: (16, 7))
        The figure size.

    fontsize : int, optional (default: 10)
        The font size for the labels and legend.

    violin_cut : float, optional (default: 0)
        The cut parameter for the violin plot. Distance, in units of bandwidth,
         to extend the density past extreme datapoints. Set to 0 to limit the
         violin within the data range.

    palette : list, optional (default: ['coral', 'lightsteelblue'])
        The color palette for the plot. Each color corresponds to a group or condition.

    lgd_bbox_to_anchor : tuple, optional (default: (1.05, 1))
        The bbox_to_anchor parameter for the legend.

    lgd_loc : str, optional (default: 'upper left')
        The location of the legend.

    save : str, optional (default: None)
        The file path to save the plot. If None, the plot is not saved.

    dpi : int, optional (default: 300)
        The resolution of the saved figure.

    tight_layout : bool, optional (default: True)
        Whether to use tight layout for the plot.

    Returns
    -------
    fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        The matplotlib Figure and Axes objects for the plot.
    """
    # Filter cells for each group
    cells_group1 = adata.obs.loc[(adata.obs[condition_key] == group1) & (adata.obs[cell_type_key] == celltype)].index
    cells_group2 = adata.obs.loc[(adata.obs[condition_key] == group2) & (adata.obs[cell_type_key] == celltype)].index

    # Compile data for significant features
    feature_dfs = []
    for feature in significant_features:
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
    sns.violinplot(data=feature_df, x='Feature', y=ylabel, hue='Group', cut=violin_cut,
                   split=True, inner="quart", linewidth=0.5, density_norm="width",
                   palette=palette, order=significant_features, ax=ax)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90, rotation_mode='anchor', ha='right', va='center',
                       fontsize=fontsize - 4)
    ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
    plt.legend(frameon=False, fontsize=fontsize - 2, bbox_to_anchor=lgd_bbox_to_anchor, loc=lgd_loc)

    if title is None:
        title = f"Feature Comparison: {group1} vs {group2} ({celltype})"
    wrapped_title = "\n".join(textwrap.wrap(title, width=wrapped_title_length))
    plt.title(wrapped_title, fontsize=fontsize + 2)

    if tight_layout:
        plt.tight_layout()

    if save:
        from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir
        dir, basename = _get_file_dir(save)
        os.makedirs(dir, exist_ok=True)
        format = _get_file_format(save)
        plt.savefig(f'{dir}/violin_{basename}.{format}', dpi=dpi, bbox_inches='tight')

    return fig, ax


def create_beeswarm_plot(df, x='log2FC', y='cell_type', cohen_threshold=0.5, pval_threshold=0.05, show_n_significant=True,
                         logfc_threshold=1.0, title=None, title_fontsize=20, ticks_fontsize=14, labels_fontsize=16,
                         condition1_color='#8B0000', condition2_color='#000080', ns_color='#808080',
                         strip_size=4, strip_alpha=0.6, strip_jitter=0.2, lgd_fontsize=14,
                         lgd_marker_size=12, lgd_frameon=False, lgd_loc='upper left', lgd_bbox_to_anchor=(1.1, 1),
                         sort_lambda=None, figsize=(10, 12), save=None, dpi=300, tight_layout=True):
    """
    Creates a beeswarm plot to visualize differential analysis results.
    X-axis represents the effect size (e.g., log2 fold change or Cohen's d).
    Y-axis represents the cell types or any categorical variable.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the results of the differential analysis, using the function `pairwise_differential_analysis`.
        The DataFrame should have at lest the following columns: 'cell_type', 'feature', 'group1',
        'group2', 'log2FC', 'cohens_d', 'adj_p_value'.

    x : str, optional (default: 'log2FC')
        The column in df to use as the x-axis.

    y : str, optional (default: 'cell_type')
        The column in df to use as the y-axis.

    cohen_threshold : float, optional (default: 0.5)
        The threshold for Cohen's D to consider a feature significant.

    pval_threshold : float, optional (default: 0.05)
        The threshold for the adjusted p-value to consider a feature significant.

    show_n_significant : bool, optional (default: True)
        Whether to show the count of significant features per cell type.

    logfc_threshold : float, optional (default: 1.0)
        The threshold for the log2 fold change to consider a feature significant.

    title : str, optional (default: None)
        The title for the plot. If None, a default title is used.

    title_fontsize : int, optional (default: 20)
        The font size for the title.

    ticks_fontsize : int, optional (default: 14)
        The font size for the ticks.

    labels_fontsize : int, optional (default: 16)
        The font size for the labels.

    condition1_color : str, optional (default: '#8B0000')
        The color for the first condition.

    condition2_color : str, optional (default: '#000080')
        The color for the second condition.

    ns_color : str, optional (default: '#808080')
        The color for non-significant features.

    strip_size : int, optional (default: 4)
        The size of the strip plot points.

    strip_alpha : float, optional (default: 0.6)
        The transparency of the strip plot points.

    strip_jitter : float, optional (default: 0.2)
        The amount of jitter to apply to the strip plot points.

    lgd_fontsize : int, optional (default: 14)
        The font size for the legend.

    lgd_marker_size : int, optional (default: 12)
        The size of the legend markers.

    lgd_frameon : bool, optional (default: False)
        Whether to show the legend frame.

    lgd_loc : str, optional (default: 'upper left')
        The location of the legend.

    lgd_bbox_to_anchor : tuple, optional (default: (1.1, 1))
        The bbox_to_anchor parameter for the legend.

    sort_lambda : function, optional (default: None)
        A lambda function to sort the y-axis values. If None, the values are sorted by the y-axis column.

    figsize : tuple, optional (default: (10, 12))
        The figure size.

    save : str, optional (default: None)
        The file path to save the plot. If None, the plot is not saved.

    dpi : int, optional (default: 300)
        The resolution of the saved figure.

    tight_layout : bool, optional (default: True)
        Whether to use tight layout for the plot.

    Returns
    -------
    fig, ax : matplotlib.pyplot.Figure, matplotlib.pyplot.Axes
        The matplotlib Figure and Axes objects for the plot.

    sig_df : DataFrame
        The input DataFrame filtered to only include significant features.
        Index is set to 'cell_type' and 'feature'.
    """
    df = df.reset_index().sort_values(by=y, key=sort_lambda)
    sorted_cells = df[y].drop_duplicates().tolist()

    # Extract conditions from contrast
    first_condition = df['group1'].iloc[0]
    second_condition = df['group2'].iloc[0]

    # Create significance column
    df['significance'] = 'NS'  # NS = Not Significant
    mask = (df['cohens_d'].abs() >= cohen_threshold) & (df['adj_p_value'] < pval_threshold)

    # Mark points as significant based on direction of change
    df.loc[mask & (df['log2FC'] <= -logfc_threshold), 'significance'] = first_condition
    df.loc[mask & (df['log2FC'] >= logfc_threshold), 'significance'] = second_condition

    # Color mapping
    color_map = {
        first_condition: condition1_color,
        second_condition: condition2_color,
        'NS': ns_color
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Add strip plot for all features
    sns.stripplot(data=df,
                  y=y,
                  x=x,
                  hue='significance',
                  palette=color_map,
                  size=strip_size,
                  alpha=strip_alpha,
                  jitter=strip_jitter,
                  ax=ax,
                  legend=False)  # Remove automatic legend

    # Add reference lines
    lines = {'cohens_d': cohen_threshold, 'log2FC': logfc_threshold}
    ax.axvline(x=-lines[x], color=condition1_color, linestyle='--', alpha=0.5)  # First condition threshold
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)  # Zero line
    ax.axvline(x=lines[x], color=condition2_color, linestyle='--', alpha=0.5)  # Second condition threshold

    # Customize the plot
    label = {'cohens_d': "Cohen's D", 'log2FC': 'log2(Fold Change)'}
    ax.set_xlabel(label[x], fontsize=labels_fontsize, fontweight='bold')
    ax.set_ylabel(y.capitalize().replace('_', ' '), fontsize=labels_fontsize, fontweight='bold')

    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[first_condition],
               markersize=lgd_marker_size,
               label=f"FDR < {pval_threshold} ({first_condition})"
               # and Cohen's D $\geq$ $\pm${cohen_threshold} ({first_condition})"
               ),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[second_condition],
               markersize=lgd_marker_size,
               label=f"FDR < {pval_threshold} ({second_condition})"
               # and Cohen's D $\geq$ $\pm${cohen_threshold} ({second_condition})"
               ),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map['NS'],
               markersize=lgd_marker_size,
               label=f"FDR $\geq$ {pval_threshold}"  # or Cohen's D < $\pm${cohen_threshold}"
               )]

    # Add custom legend
    legend = ax.legend(handles=legend_elements,
                       bbox_to_anchor=lgd_bbox_to_anchor,
                       fontsize=lgd_fontsize,
                       frameon=lgd_frameon,
                       loc=lgd_loc)

    # Show all spines
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    xlim = ax.get_xlim()

    # Add count of significant features per cell type
    if show_n_significant:
        for i, cell_type in enumerate(sorted_cells):
            cell_data = df[df[y] == cell_type]
            sig_first = sum((cell_data['significance'] == first_condition))
            sig_second = sum((cell_data['significance'] == second_condition))
            if sig_first > 0 or sig_second > 0:
                ax.text(xlim[1] + 0.1, i, f'n={sig_first + sig_second}',
                        va='center', ha='left', fontsize=ticks_fontsize, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
    # Add contrast as title
    if title is None:
        title = f'{first_condition} vs {second_condition}'
    plt.title(title, pad=20, fontweight='bold', fontsize=title_fontsize)

    plt.grid(color='#E0E0E0')

    # Adjust layout
    if tight_layout:
        plt.tight_layout()

    # Output
    sig_df = df.loc[df.significance != 'NS'].drop(columns=['significance']).reset_index(drop=True).set_index(['cell_type', 'feature'])

    if save:
        from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir
        dir, basename = _get_file_dir(save)
        os.makedirs(dir, exist_ok=True)
        format = _get_file_format(save)
        plt.savefig(f'{dir}/beeswarm_{basename}.{format}', dpi=dpi, bbox_inches='tight')
    return fig, ax, sig_df