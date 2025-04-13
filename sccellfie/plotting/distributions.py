import os
import textwrap
import scanpy as sc
import matplotlib.pyplot as plt


def create_multi_violin_plots(adata, features, groupby, n_cols=4, figsize=(5, 5), ylabel=None, title=None, fontsize=10,
                              rotation=90, wrapped_title_length=45, save=None, dpi=300, tight_layout=True, w_pad=None,
                              h_pad=None, **kwargs):
    """
    Plots a grid of violin plots for multiple genes in Scanpy,
    controlling the number of columns.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    features : list of str
        List of feature names to plot. Should match names in
        adata.var_names.

    groupby : str
        Key in `adata.obs` containing the groups to plot. For each
        unique value in this column, a violin plot will be generated.

    n_cols : int, optional (default: 4)
        Number of columns in the grid.

    figsize : tuple of float, optional (default: (5, 5))
        Size of each subplot in inches.

    ylabel : str, optional (default: None)
        Label for the y-axis. If None, the label will be the variable name.

    title : list of str, optional (default: None)
        List of labels for each feature. If None, the feature name will be used.

    fontsize : int, optional (default: 10)
        Font size for the title and axis labels. The tick labels will
        be set to `fontsize`, while the title will be set to `fontsize + 4`.
        Ylabel will be set to `fontsize + 2`.

    rotation : int, optional (default: 90)
        Rotation of the x-axis tick labels

    wrapped_title_length : int, optional (default: 50)
        The maximum number of characters per line in the title.

    save : str, optional (default: None)
        Filepath to save the figure. If not provided, the figure
        will be displayed.

    dpi : int, optional (default: 300)
        Resolution of the saved figure.

    tight_layout : bool, optional (default: True)
        Whether to use tight layout.

    w_pad : float, optional (default: None)
        Width padding between subplots.

    h_pad : float, optional (default: None)
        Height padding between subplots.

    **kwargs : dict
        Additional arguments to pass to `sc.pl.violin`. For example,
        `rotation` can be used to rotate the x-axis labels.
    """
    n_genes = len(features)
    n_rows = -(-n_genes // n_cols)  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
                             squeeze=False)
    fig.tight_layout(pad=3.0)

    for i, feature in enumerate(features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        sc.pl.violin(adata, keys=feature, groupby=groupby, ax=ax, show=False, rotation=rotation, **kwargs)
        if title is not None:
            wrapped_title = "\n".join(textwrap.wrap(title[i], width=wrapped_title_length))
        else:
            wrapped_title = "\n".join(textwrap.wrap(feature, width=wrapped_title_length))

        if ylabel is None:
            ylabel_ = wrapped_title
        else:
            ax.set_title(wrapped_title, fontsize=fontsize + 4)
            ylabel_ = ylabel
        ax.set_ylabel(ylabel_, fontsize=fontsize + 2)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

    # Remove empty subplots
    for i in range(n_genes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    if tight_layout:
        plt.tight_layout(w_pad=w_pad, h_pad=h_pad)

    if save:
        from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir
        dir, basename = _get_file_dir(save)
        os.makedirs(dir, exist_ok=True)
        format = _get_file_format(save)
        plt.savefig(f'{dir}/violin_{basename}.{format}', dpi=dpi, bbox_inches='tight')
    return fig, axes