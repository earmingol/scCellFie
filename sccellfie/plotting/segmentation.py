"""
Cell-level spatial visualization for AnnData objects.

Provides :func:`plot_segmentation`, a general-purpose renderer for
cell polygons (from segmentation) or centroid scatter plots, with
categorical and continuous colouring, customizable legends, optional
crop, and a scalebar. Works for any technology that exposes per-cell
2D coordinates (Xenium, VisiumHD-segmented, Atera, ...).
"""
import math
import textwrap
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.transforms import blended_transform_factory, offset_copy


def _add_scalebar(
    ax,
    length=None,
    units="µm",
    color="black",
    linewidth=3,
    fontsize=9,
    position="lower_right",
    pad_frac=0.05,
    text_pad_pts=2.0,
    remove_axes=True,
):
    """Add a scalebar to a spatial axes (1-2-5 auto-snap, inversion-aware).

    Uses a blended transform (data X, axes-fraction Y) so the bar position is
    independent of data extent or figure size: ``pad_frac`` is the fraction of
    the axes height/width inset from the chosen corner. The label is placed on
    the side away from the data (below the bar for ``lower_*``, above for
    ``upper_*``) so it never overlaps the cells, regardless of ``invert_yaxis``.
    """
    xlim = ax.get_xlim()
    width = abs(xlim[1] - xlim[0])

    if length is None:
        raw = 0.2 * width
        if raw > 0:
            exponent = int(math.floor(math.log10(raw)))
            base = raw / (10 ** exponent)
            nice_base = 5 if base > 5 else 2 if base > 2 else 1
            length = nice_base * (10 ** exponent)
        else:
            length = 1.0

    if "right" in position:
        x_end = xlim[1] - pad_frac * width
        x_start = x_end - length
    else:
        x_start = xlim[0] + pad_frac * width
        x_end = x_start + length

    is_lower = "lower" in position
    y_axes = pad_frac if is_lower else 1.0 - pad_frac

    blended = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        [x_start, x_end], [y_axes, y_axes],
        color=color, lw=linewidth, solid_capstyle="butt",
        transform=blended, clip_on=False,
    )
    label = f"{int(length) if length >= 1 else length} {units}"

    sign = -1.0 if is_lower else 1.0
    text_transform = offset_copy(
        blended, fig=ax.figure,
        y=sign * (linewidth / 2.0 + text_pad_pts),
        units="points",
    )
    text_va = "top" if is_lower else "bottom"
    ax.text(
        (x_start + x_end) / 2.0, y_axes, label,
        color=color, transform=text_transform,
        ha="center", va=text_va, fontsize=fontsize,
        clip_on=False,
    )

    if remove_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)


def _render_panel(
    ax,
    *,
    adata,
    local_idx,
    cell_ids,
    coords,
    xlim,
    ylim,
    color_by,
    celltype_key,
    segmentation,
    palette,
    highlight,
    layer,
    legend,
    legend_loc,
    legend_bbox,
    legend_frameon,
    legend_title,
    legend_fontsize,
    legend_ncol,
    legend_params,
    axes_off,
    scatter_size,
    cmap,
    scalebar,
    scalebar_kwargs,
    cbar_kwargs,
    panel_title,
    title_fontsize,
):
    """Render a single panel (one feature) into ``ax``."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_autoscale_on(False)

    target_col = color_by if color_by is not None else celltype_key
    colors_array = None
    cmap_vals = None
    is_categorical = False
    final_palette: dict = {}

    if target_col in adata.obs.columns:
        col_data = adata.obs[target_col].iloc[local_idx]
        if isinstance(col_data.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(col_data):
            is_categorical = True
            unique_cats = list(adata.obs[target_col].astype("category").cat.categories)

            if palette:
                final_palette = dict(palette)
            elif f"{target_col}_colors" in adata.uns:
                final_palette = dict(zip(unique_cats, adata.uns[f"{target_col}_colors"]))
            else:
                s2 = plt.get_cmap("Set2")
                final_palette = {cat: s2(i % 8) for i, cat in enumerate(unique_cats)}

            if highlight:
                h_set = set(highlight)
                for cat in unique_cats:
                    if cat not in h_set:
                        final_palette[cat] = "whitesmoke"

            colors_array = [final_palette.get(v, "whitesmoke") for v in col_data.values]
        else:
            cmap_vals = col_data.values
    elif target_col in adata.var_names:
        sub = adata[local_idx, target_col]
        gene_data = sub.layers[layer] if layer is not None else sub.X
        if scipy.sparse.issparse(gene_data):
            cmap_vals = gene_data.toarray().flatten()
        else:
            cmap_vals = np.asarray(gene_data).flatten()
    else:
        colors_array = ["#cccccc"] * len(local_idx)

    p_coll = None
    scatter = None
    if segmentation is not None:
        patches, v_cols, v_cmap = [], [], []
        for i, idx in enumerate(local_idx):
            cid = cell_ids[idx]
            poly = segmentation.get(cid)
            if poly is None:
                continue
            patches.append(MplPolygon(np.array(poly.exterior.coords), closed=True))
            if colors_array is not None:
                v_cols.append(colors_array[i])
            if cmap_vals is not None:
                v_cmap.append(cmap_vals[i])

        p_coll = PatchCollection(
            patches, alpha=1.0, edgecolor="gray", linewidth=0.05, antialiased=True
        )
        if colors_array is not None:
            p_coll.set_facecolor(v_cols)
        else:
            p_coll.set_array(np.array(v_cmap))
            p_coll.set_cmap(cmap)
        ax.add_collection(p_coll)
    else:
        l_coords = coords[local_idx]
        c = colors_array if colors_array is not None else cmap_vals
        scatter = ax.scatter(
            l_coords[:, 0],
            l_coords[:, 1],
            c=c,
            cmap=cmap if cmap_vals is not None else None,
            s=scatter_size,
            edgecolor="none",
        )

    if legend:
        if is_categorical:
            if highlight:
                items = [
                    (cat, col)
                    for cat, col in final_palette.items()
                    if col != "whitesmoke"
                ]
            else:
                items = list(final_palette.items())
            handles = [mpatches.Patch(color=col, label=cat) for cat, col in items]
            if handles:
                kw = dict(
                    handles=handles,
                    loc=legend_loc,
                    frameon=legend_frameon,
                    ncol=legend_ncol,
                )
                if legend_bbox is not None:
                    kw["bbox_to_anchor"] = legend_bbox
                if legend_title is not None:
                    kw["title"] = legend_title
                if legend_fontsize is not None:
                    kw["fontsize"] = legend_fontsize
                if legend_params:
                    kw.update(legend_params)
                ax.legend(**kw)
        elif cmap_vals is not None:
            mappable = p_coll if p_coll is not None else scatter
            cb_kw = dict(fraction=0.046, pad=0.04)
            if cbar_kwargs:
                cb_kw.update(cbar_kwargs)
            plt.colorbar(mappable, ax=ax, **cb_kw)

    ax.set_aspect("equal")

    if scalebar:
        sb_params = {"color": "black", "position": "lower_right", "pad_frac": 0.05}
        if scalebar_kwargs:
            sb_params.update(scalebar_kwargs)
        _add_scalebar(ax, remove_axes=axes_off, **sb_params)
    elif axes_off:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Final safeguard: re-apply limits after every artist has been added,
    # in case anything (notably colorbar or scalebar) caused a relim.
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    if panel_title is not None:
        ax.set_title(panel_title, fontsize=title_fontsize)


def plot_segmentation(
    adata,
    spatial_key: str = "X_spatial",
    color_by: Optional[Union[str, Sequence[str]]] = None,
    celltype_key: str = "cell_type",
    segmentation: Optional[dict] = None,
    cell_id_col: Optional[str] = None,
    palette: Optional[dict] = None,
    highlight: Optional[List[str]] = None,
    layer: Optional[str] = None,
    crop: Optional[Tuple[float, float, float, float]] = None,
    invert_yaxis: bool = True,
    legend: bool = True,
    legend_loc: str = "center left",
    legend_bbox: Optional[Tuple[float, float]] = (1.01, 0.5),
    legend_frameon: bool = False,
    legend_title: Optional[str] = None,
    legend_fontsize: Optional[float] = 7.0,
    legend_ncol: int = 1,
    legend_params: Optional[dict] = None,
    axes_off: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ax=None,
    ncols: int = 4,
    panel_titles: bool = True,
    title: Optional[Union[str, Sequence[str]]] = None,
    title_fontsize: Optional[float] = 12,
    wrapped_title_length: int = 45,
    dpi: int = 150,
    scatter_size: float = 2.0,
    cmap: str = "viridis",
    y_pad_ratio: float = 0.1,
    x_pad_ratio: float = 0.0,
    scalebar: bool = True,
    scalebar_kwargs: Optional[dict] = None,
    cbar_kwargs: Optional[dict] = None,
    save: Optional[str] = None,
):
    """
    Plot cell-resolution spatial data from an AnnData object.

    Renders cells as segmentation polygons when ``segmentation`` is
    provided, otherwise as a centroid scatter plot. Supports categorical
    and continuous colouring, optional highlighting of a subset of
    categories, axis cropping, and a scalebar with bottom/top padding.

    When ``color_by`` is a list, multiple panels are drawn in a grid
    laid out by ``ncols`` (matching ``sc.pl.spatial`` semantics): the
    geometry, crop, and view limits are computed once and shared across
    panels; each panel is coloured independently and gets its own legend
    or colorbar.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with spatial coordinates in ``adata.obsm[spatial_key]``.

    spatial_key : str, optional (default: "X_spatial")
        Key in ``adata.obsm`` for the ``(n_cells, 2+)`` coordinate
        array. Defaults to scCellFie's canonical key.

    color_by : str, list of str, or None, optional (default: None)
        Column in ``adata.obs`` or name in ``adata.var_names`` to colour
        by. If None, falls back to ``celltype_key``. Pass a list of
        names (e.g. ``["task_A", "task_B", "GENE1"]``) to render a
        multi-panel figure with one panel per feature.

    celltype_key : str, optional (default: "cell_type")
        Default categorical column used when ``color_by`` is None.

    segmentation : dict, optional (default: None)
        Mapping ``cell_id -> shapely.Polygon`` (e.g. output of
        :func:`sccellfie.io.load_segmentation` with ``output="dict"``).
        If None, a scatter of centroids is drawn.

    cell_id_col : str, optional (default: None)
        Column in ``adata.obs`` identifying cells. Defaults to
        ``adata.obs.index``.

    palette : dict, optional (default: None)
        Custom ``{category: color}`` mapping for categorical colouring.
        Falls back to ``adata.uns["{color_by}_colors"]`` or matplotlib
        ``Set2`` cycling. In multi-panel mode the same palette is reused
        for every categorical feature.

    highlight : list of str, optional (default: None)
        Subset of categories to highlight; all others are drawn in
        ``whitesmoke`` and excluded from the legend.

    layer : str, optional (default: None)
        Layer name in ``adata.layers`` used when ``color_by`` is a gene.
        If None, uses ``adata.X``.

    crop : tuple, optional (default: None)
        ``(minx, miny, maxx, maxy)`` bounds to restrict the view. Data
        outside this box is not rendered. If None, uses data extent.

    invert_yaxis : bool, optional (default: True)
        Invert the y-axis (microscopy convention).

    legend : bool, optional (default: True)
        Show the legend for categorical data, or a colorbar for
        continuous data.

    legend_loc : str, optional (default: "center left")
        ``loc`` argument passed to ``ax.legend()``. Ignored for colorbar.

    legend_bbox : tuple, optional (default: (1.01, 0.5))
        ``bbox_to_anchor`` for the legend. Use ``None`` to disable the
        anchor and rely on ``legend_loc`` alone.

    legend_frameon : bool, optional (default: False)
        Whether the legend frame/border is drawn.

    legend_title : str, optional (default: None)
        Title shown above the legend entries.

    legend_fontsize : float, optional (default: 7.0)
        Font size for legend labels. ``None`` falls back to the matplotlib
        default. The small default suits spatial plots with many
        categories; bump it via ``legend_params={'fontsize': 10}`` (or
        the dedicated arg) when needed.

    legend_ncol : int, optional (default: 1)
        Number of columns in the legend.

    legend_params : dict, optional (default: None)
        Arbitrary kwargs forwarded to ``ax.legend(...)`` (e.g.
        ``handlelength``, ``labelspacing``, ``borderpad``,
        ``columnspacing``). Keys here override the dedicated
        ``legend_*`` arguments on conflict.

    axes_off : bool, optional (default: True)
        Remove ticks, tick labels, and spines (standard for spatial plots).

    figsize : tuple, optional (default: None)
        - Single panel (``color_by`` is a str or None): the figure size,
          defaulting to ``(10, 10)`` when None.
        - Multi panel (``color_by`` is a list): the per-panel size,
          defaulting to ``(4, 4)`` when None. The total figure size is
          ``(figsize[0] * ncols, figsize[1] * nrows)``.

        Ignored when ``ax`` is provided.

    ax : matplotlib.axes.Axes, optional (default: None)
        Existing axes to draw onto. Only valid when ``color_by`` is a
        single feature (or None). For multi-panel, omit ``ax`` and let
        the function build the grid.

    ncols : int, optional (default: 4)
        Number of columns in the panel grid when ``color_by`` is a list.
        Number of rows is ``ceil(len(color_by) / ncols)``. Mirrors
        ``sc.pl.spatial``'s ``ncols`` parameter.

    panel_titles : bool, optional (default: True)
        Master toggle for panel titles. When True, each panel's title is
        set to the corresponding feature name (or to the explicit string
        passed via ``title=``). Set False to suppress titles entirely
        (in single- and multi-panel modes).

    title : str, list of str, or None, optional (default: None)
        Explicit title override. For single-feature mode pass a string;
        for multi-feature mode pass a list of strings whose length
        matches ``color_by``. When None (default), titles are auto-derived
        from the feature names. Ignored if ``panel_titles=False``.

    title_fontsize : float, optional (default: 12)
        Font size of the per-panel title. Mirrors the convention in
        :func:`sccellfie.plotting.plot_spatial`.

    wrapped_title_length : int, optional (default: 45)
        Maximum number of characters per title line. Long feature names
        (e.g. metabolic-task labels) are wrapped via :func:`textwrap.wrap`
        before being set, matching the behavior of the other tool plots
        (:func:`plot_spatial`, :func:`create_multi_violin_plots`,
        :func:`create_volcano_plot`). Pass a large value (e.g. 1000) to
        disable wrapping.

    dpi : int, optional (default: 150)
        Figure DPI; also used when ``save`` is set.

    scatter_size : float, optional (default: 2.0)
        Marker size for centroid scatter mode.

    cmap : str, optional (default: "viridis")
        Matplotlib colormap name for continuous colouring.

    y_pad_ratio : float, optional (default: 0.1)
        Fraction of the y range added as top/bottom whitespace (so the
        scalebar label has room).

    x_pad_ratio : float, optional (default: 0.0)
        Fraction of the x range added as left/right whitespace. Default
        keeps x tight to data — increase when the legend or a colorbar
        sits to the right of the plot and you want extra breathing room
        on the data side too.

    scalebar : bool, optional (default: True)
        Draw a scalebar on every panel.

    scalebar_kwargs : dict, optional (default: None)
        Overrides for the scalebar (e.g. ``length``, ``units``, ``color``,
        ``position``, ``pad_frac``, ``fontsize``, ``text_pad_pts``).
        ``pad_frac`` is the inset of the bar from the axes corner as a
        fraction of the axes height/width; ``text_pad_pts`` is the gap (in
        points) between the bar and its label. The label is always placed
        on the side of the bar away from the data, so it never overlaps
        cells when ``y_pad_ratio > 0``.

    cbar_kwargs : dict, optional (default: None)
        Overrides passed to ``plt.colorbar`` for continuous colouring.

    save : str, optional (default: None)
        If given, save the figure to this path with ``dpi`` and
        ``bbox_inches="tight"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    ax : matplotlib.axes.Axes or numpy.ndarray of Axes
        Single Axes when ``color_by`` is a string (or None); a 2D array
        of Axes (shape ``(nrows, ncols)``) when ``color_by`` is a list.
    """
    is_list = isinstance(color_by, (list, tuple)) and not isinstance(color_by, str)
    features = list(color_by) if is_list else [color_by]
    n_panels = len(features)

    if is_list and ax is not None:
        raise ValueError(
            "`ax` is only supported for a single `color_by`. "
            "When passing a list of features, omit `ax` so the grid can be built."
        )

    if is_list:
        nrows_grid = math.ceil(n_panels / ncols)
        ncols_grid = ncols
        per_panel = figsize if figsize is not None else (4, 4)
        total_figsize = (per_panel[0] * ncols_grid, per_panel[1] * nrows_grid)
    else:
        total_figsize = figsize if figsize is not None else (10, 10)
        nrows_grid, ncols_grid = 1, 1

    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    cell_ids = (
        adata.obs[cell_id_col].values
        if cell_id_col is not None
        else adata.obs.index.values
    )

    if crop is not None:
        d_minx, d_miny, d_maxx, d_maxy = crop
    else:
        d_minx, d_maxx = coords[:, 0].min(), coords[:, 0].max()
        d_miny, d_maxy = coords[:, 1].min(), coords[:, 1].max()

    mask = (
        (coords[:, 0] >= d_minx)
        & (coords[:, 0] <= d_maxx)
        & (coords[:, 1] >= d_miny)
        & (coords[:, 1] <= d_maxy)
    )
    local_idx = np.where(mask)[0]

    xr = d_maxx - d_minx
    yr = d_maxy - d_miny
    xp = xr * x_pad_ratio
    yp = yr * y_pad_ratio
    xlim = (d_minx - xp, d_maxx + xp)
    ylim = (d_maxy + yp, d_miny - yp) if invert_yaxis else (d_miny - yp, d_maxy + yp)

    if ax is None:
        fig, axes = plt.subplots(
            nrows_grid, ncols_grid,
            figsize=total_figsize, dpi=dpi, squeeze=False,
        )
    else:
        fig = ax.get_figure()
        axes = np.array([[ax]])

    flat_axes = axes.ravel()

    if panel_titles:
        if title is not None:
            if is_list:
                if not isinstance(title, (list, tuple)) or isinstance(title, str):
                    raise ValueError(
                        "`title` must be a list/tuple of strings when `color_by` is a list."
                    )
                if len(title) != n_panels:
                    raise ValueError(
                        f"`title` has {len(title)} entries but `color_by` has {n_panels}."
                    )
                resolved_titles = list(title)
            else:
                if not isinstance(title, str):
                    raise ValueError(
                        "`title` must be a string when `color_by` is a single feature."
                    )
                resolved_titles = [title]
        else:
            resolved_titles = [
                (f if f is not None else celltype_key) for f in features
            ]
    else:
        resolved_titles = [None] * n_panels

    for i, feature in enumerate(features):
        raw = resolved_titles[i]
        if raw is None:
            wrapped = None
        else:
            wrapped = "\n".join(textwrap.wrap(str(raw), width=wrapped_title_length))
            if wrapped == "":
                wrapped = str(raw)
        _render_panel(
            flat_axes[i],
            adata=adata,
            local_idx=local_idx,
            cell_ids=cell_ids,
            coords=coords,
            xlim=xlim,
            ylim=ylim,
            color_by=feature,
            celltype_key=celltype_key,
            segmentation=segmentation,
            palette=palette,
            highlight=highlight,
            layer=layer,
            legend=legend,
            legend_loc=legend_loc,
            legend_bbox=legend_bbox,
            legend_frameon=legend_frameon,
            legend_title=legend_title,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_params=legend_params,
            axes_off=axes_off,
            scatter_size=scatter_size,
            cmap=cmap,
            scalebar=scalebar,
            scalebar_kwargs=scalebar_kwargs,
            cbar_kwargs=cbar_kwargs,
            panel_title=wrapped,
            title_fontsize=title_fontsize,
        )

    for j in range(n_panels, flat_axes.size):
        flat_axes[j].set_visible(False)

    if is_list:
        fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    if is_list:
        return fig, axes
    return fig, flat_axes[0]
