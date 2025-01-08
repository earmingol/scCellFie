import textwrap
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_spatial(adata, keys, suptitle=None, bkgd_label='H&E', ncols=3, figsize=(5, 5), save=None, dpi=300, bbox_inches='tight'):
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

    bkgd_label : str, optional (default: 'H&E')
        Label for the background image.

    ncols : int, optional (default: 3)
        Number of columns in the grid.

    figsize : tuple of float, optional (default: (5, 5))
        Size of each subplot in inches.

    save : str, optional (default: None)
        Filepath to save the figure.

    dpi : int, optional (default: 300)
        Resolution of the saved figure. Only used if `save` is provided.

    bbox_inches : str, optional (default: 'tight')
        Bounding box in inches. Only used if `save` is provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    axes : numpy.ndarray
        Array of matplotlib axes.
    """
    n_genes = len(keys)
    n_cols = min([ncols, n_genes])
    n_rows = -(-n_genes // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
                             squeeze=False)
    fig.tight_layout(pad=3.0)
    for i, gene in enumerate(keys):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        sc.pl.spatial(adata, img_key="hires", color=gene,
                      use_raw=False, cmap='YlGnBu', size=1.2, ncols=3, vmin=0, ax=ax, show=False)
        if gene is not None:
            wrapped_title = "\n".join(textwrap.wrap(gene, width=40))
            ax.set_title(wrapped_title, fontsize=12)
        else:
            ax.set_title(bkgd_label, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Remove empty subplots
    for i in range(n_genes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    fig.suptitle(suptitle, y=1.0, fontsize=20)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches=bbox_inches)

    return fig, axes


def plot_neighbor_distribution(results, figsize=(15, 8), save=None, dpi=300, bbox_inches='tight'):
    """
    Visualizes the neighbor distribution analysis results.

    Parameters
    ----------
    results : dict
        Output from ´sccellfie.spatial.neighborhood.compute_neighbor_distribution´ function

    figsize : tuple
        Figure size for the combined plots

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

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi, bbox_inches=bbox_inches)
    return fig, gs