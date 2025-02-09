import os
import textwrap
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_spatial(adata, keys, suptitle=None, suptitle_fontsize=20, title_fontsize=14, legend_fontsize=12,
                 bkgd_label='H&E', wrapped_title_length=45, ncols=3, hspace=0.15, wspace=0.1, save=None, dpi=300,
                 bbox_inches='tight', tight_layout=True, **kwargs):
    """
    Plots spatial expression of multiple genes in Scanpy.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression and spatial information.

    keys : list of str
        List of feature names to plot. Should match names in
        adata.var_names or a column in adata.obs.

    suptitle : str, optional (default: None)
        Title for the entire figure.

    suptitle_fontsize : int, optional (default: 20)
        Font size for the figure title.

    title_fontsize : int, optional (default: 14)
        Font size for each subplot title (key name).

    legend_fontsize : int, optional (default: 12)
        Font size for the legend elements.

    hspace : float, optional (default: 0.1)
        Height space between subplots.

    wspace : float, optional (default: 0.1)
        Width space between subplots.

    bkgd_label : str, optional (default: 'H&E')
        Label for the background image.

    wrapped_title_length : int, optional (default: 45)
        The maximum number of characters per line in the title.

    ncols : int, optional (default: 3)
        Number of columns in the grid.

    save : str, optional (default: None)
        Filepath to save the figure.

    dpi : int, optional (default: 300)
        Resolution of the saved figure. Only used if `save` is provided.

    bbox_inches : str, optional (default: 'tight')
        Bounding box in inches. Only used if `save` is provided.

    tight_layout : bool, optional (default: True)
        Whether to use tight layout.

    **kwargs : dict
        Additional arguments to pass to `scanpy.pl.spatial`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    axes : numpy.ndarray
        Array of matplotlib axes.
    """
    n_genes = len(keys)
    n_cols = min([ncols, n_genes])

    titles = []
    if 'title' in kwargs.keys():
        if kwargs['title'] is None:
            title = keys
            del kwargs['title']
        else:
            title = kwargs.pop('title')
    else:
        title = keys
    for i, gene in enumerate(title):
        if gene is not None:
            wrapped_title = "\n".join(textwrap.wrap(gene, width=wrapped_title_length))
            titles.append(wrapped_title)
        else:
            titles.append(bkgd_label)

    axes = sc.pl.spatial(adata, color=keys, ncols=n_cols, hspace=hspace, wspace=wspace, title=titles,
                         legend_fontsize=legend_fontsize, show=False, **kwargs)
    for title, ax in zip(titles, axes):
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig = plt.gcf()

    fig.suptitle(suptitle, y=1.05, fontsize=suptitle_fontsize)

    if tight_layout:
        plt.tight_layout()

    if save:
        from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir
        dir, basename = _get_file_dir(save)
        os.makedirs(dir, exist_ok=True)
        format = _get_file_format(save)
        plt.savefig(f'{dir}/spatial_{basename}.{format}', dpi=dpi, bbox_inches=bbox_inches)

    return fig, axes


def plot_neighbor_distribution(results, figsize=(15, 8), save=None, dpi=300, bbox_inches='tight', tight_layout=True):
    """
    Visualizes the neighbor distribution analysis results.

    Parameters
    ----------
    results : dict
        Output from ´sccellfie.spatial.neighborhood.compute_neighbor_distribution´ function

    figsize : tuple
        Figure size for the combined plots

    save : str, optional (default: None)
        Filepath to save the figure.

    dpi : int, optional (default: 300)
        Resolution of the saved figure.

    bbox_inches : str, optional (default: 'tight')
        Bounding box in inches. Only used if `save` is provided.

    tight_layout : bool, optional (default: True)
        Whether to use tight layout.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    gs : matplotlib.gridspec.GridSpec
        The matplotlib gridspec object.
    """
    radii = results['radii']
    mean_neighbors = results['mean']
    neighbor_counts = results['neighbors']
    quantiles = list(results['quantiles'].keys())
    quantile_values = np.array(list(results['quantiles'].values()))

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig)

    # Plot 1: Quantile ranges and mean (larger plot)
    ax1 = fig.add_subplot(gs[0, :])

    # Plot quantile ranges

    ax1.fill_between(radii,
                     quantile_values[0],
                     quantile_values[-1],
                     alpha=0.2)

    # Plot mean
    ax1.plot(radii, mean_neighbors, 'r-', label='Mean', linewidth=2)

    # Add labels and title
    ax1.set_xlabel('Radius')
    ax1.set_ylabel('Number of Neighbors')
    ax1.set_title('Distribution of Neighbors per Spot vs Radius')

    # Add legend for quantiles
    ax1.plot([], [], 'b-', alpha=0.2, label='{}% CI'.format(int(max(quantiles) * 100) - int(min(quantiles) * 100)))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2-4: Histograms at specific radii
    n_examples = 3
    example_radii = np.linspace(radii[0], radii[-1], n_examples + 2)[1:-1]

    for i, radius in enumerate(example_radii):
        ax = fig.add_subplot(gs[1, i])
        idx = np.argmin(np.abs(radii - radius))
        sns.histplot(neighbor_counts[:, idx], ax=ax)
        ax.set_title(f'Radius = {radius:.1f}')
        ax.set_xlabel('Number of neighbors')

    if tight_layout:
        plt.tight_layout()

    if save:
        from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir
        dir, basename = _get_file_dir(save)
        os.makedirs(dir, exist_ok=True)
        format = _get_file_format(save)
        plt.savefig(f'{dir}/spatial_{basename}.{format}', dpi=dpi, bbox_inches=bbox_inches)
    return fig, gs