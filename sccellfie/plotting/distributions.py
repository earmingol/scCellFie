import scanpy as sc
import matplotlib.pyplot as plt


def create_multi_violin_plots(adata, genes, groupby, n_cols=4, figsize=(4, 3), ylabel='Metabolic Activity', fontsize=10, save=None, dpi=300, **kwargs):
    """
    Plot a grid of violin plots for multiple genes in Scanpy,
    controlling the number of columns.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.

    genes : list of str
        List of gene names to plot.

    groupby : str
        Key in `adata.obs` containing the groups to plot. For each
        unique value in this column, a violin plot will be generated.

    n_cols : int, optional (default=4)
        Number of columns in the grid.

    figsize : tuple of float, optional (default=(4, 3))
        Size of each subplot in inches.

    ylabel : str, optional (default='Metabolic Activity')
        Label for the y-axis.

    fontsize : int, optional (default=10)
        Font size for the title and axis labels. The tick labels will
        be set to `fontsize`, while the title will be set to `fontsize + 4`.
        Ylabel will be set to `fontsize + 2`.

    save : str, optional (default=None)
        Filepath to save the figure. If not provided, the figure
        will be displayed.

    dpi : int, optional (default=300)
        Resolution of the saved figure.

    **kwargs : dict
        Additional arguments to pass to `sc.pl.violin`. For example,
        `rotation` can be used to rotate the x-axis labels.
    """
    n_genes = len(genes)
    n_rows = -(-n_genes // n_cols)  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
                             squeeze=False)
    fig.tight_layout(pad=3.0)

    for i, gene in enumerate(genes):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        sc.pl.violin(adata, keys=gene, groupby=groupby, ax=ax, show=False, **kwargs)
        ax.set_title(gene, fontsize=fontsize + 4)
        ax.set_ylabel(ylabel, fontsize=fontsize + 2)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

    # Remove empty subplots
    for i in range(n_genes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save} with DPI {dpi}")

    return fig, axes