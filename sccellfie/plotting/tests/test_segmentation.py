import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sccellfie.datasets.toy_inputs import create_controlled_adata_with_spatial
from sccellfie.plotting import plot_segmentation


def test_plot_segmentation_scatter_categorical():
    adata = create_controlled_adata_with_spatial()
    fig, ax = plot_segmentation(
        adata, color_by="group", celltype_key="group", scalebar=False
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    leg = ax.get_legend()
    assert leg is not None
    labels = [t.get_text() for t in leg.get_texts()]
    assert set(labels) == {"A", "B"}
    plt.close(fig)


def test_plot_segmentation_scatter_continuous_gene():
    adata = create_controlled_adata_with_spatial()
    fig, ax = plot_segmentation(adata, color_by="gene1", scalebar=False)
    # Continuous => colorbar should be added to the figure
    assert any(a is not ax for a in fig.axes)
    plt.close(fig)


def test_plot_segmentation_crop_filters_cells():
    adata = create_controlled_adata_with_spatial()
    fig, ax = plot_segmentation(
        adata,
        color_by="group",
        celltype_key="group",
        crop=(-0.5, -0.5, 1.5, 1.5),
        scalebar=False,
    )
    coll = ax.collections[0]
    assert coll.get_offsets().shape[0] == 2
    plt.close(fig)


def test_plot_segmentation_with_scalebar():
    adata = create_controlled_adata_with_spatial()
    fig, ax = plot_segmentation(
        adata, color_by="group", celltype_key="group", scalebar=True
    )
    assert len(ax.get_lines()) >= 1
    assert len(ax.texts) >= 1
    plt.close(fig)


def test_plot_segmentation_invert_yaxis():
    adata = create_controlled_adata_with_spatial()
    fig, ax = plot_segmentation(
        adata, color_by="group", celltype_key="group", invert_yaxis=True, scalebar=False
    )
    ylim = ax.get_ylim()
    assert ylim[0] > ylim[1]
    plt.close(fig)


def test_plot_segmentation_save(tmp_path):
    adata = create_controlled_adata_with_spatial()
    out = tmp_path / "fig.png"
    fig, ax = plot_segmentation(
        adata, color_by="group", celltype_key="group", scalebar=False, save=str(out)
    )
    assert out.exists()
    plt.close(fig)


def test_plot_segmentation_polygon_mode():
    pytest.importorskip("shapely")
    from shapely.geometry import Polygon

    adata = create_controlled_adata_with_spatial()
    seg = {}
    for i, name in enumerate(adata.obs_names):
        x, y = adata.obsm["X_spatial"][i]
        seg[name] = Polygon(
            [(x - 0.3, y - 0.3), (x + 0.3, y - 0.3), (x + 0.3, y + 0.3), (x - 0.3, y + 0.3)]
        )

    fig, ax = plot_segmentation(
        adata,
        color_by="group",
        celltype_key="group",
        segmentation=seg,
        scalebar=False,
    )
    # Polygon mode adds a PatchCollection
    from matplotlib.collections import PatchCollection

    assert any(isinstance(c, PatchCollection) for c in ax.collections)
    plt.close(fig)


def test_plot_segmentation_polygon_continuous():
    pytest.importorskip("shapely")
    from shapely.geometry import Polygon

    adata = create_controlled_adata_with_spatial()
    seg = {
        name: Polygon(
            [
                (adata.obsm["X_spatial"][i, 0] - 0.3, adata.obsm["X_spatial"][i, 1] - 0.3),
                (adata.obsm["X_spatial"][i, 0] + 0.3, adata.obsm["X_spatial"][i, 1] - 0.3),
                (adata.obsm["X_spatial"][i, 0] + 0.3, adata.obsm["X_spatial"][i, 1] + 0.3),
                (adata.obsm["X_spatial"][i, 0] - 0.3, adata.obsm["X_spatial"][i, 1] + 0.3),
            ]
        )
        for i, name in enumerate(adata.obs_names)
    }

    fig, ax = plot_segmentation(
        adata, color_by="gene1", segmentation=seg, scalebar=False
    )
    # Continuous + polygons => colorbar in fig
    assert any(a is not ax for a in fig.axes)
    plt.close(fig)


def test_plot_segmentation_highlight_subset():
    adata = create_controlled_adata_with_spatial()
    fig, ax = plot_segmentation(
        adata,
        color_by="group",
        celltype_key="group",
        highlight=["A"],
        scalebar=False,
    )
    leg = ax.get_legend()
    labels = [t.get_text() for t in leg.get_texts()]
    assert labels == ["A"]
    plt.close(fig)


def test_plot_segmentation_limits_locked_with_polygons_and_colorbar():
    """Polygon rendering + a colorbar must NOT clobber the padded y-limits."""
    pytest.importorskip("shapely")
    from shapely.geometry import Polygon

    adata = create_controlled_adata_with_spatial()
    seg = {
        name: Polygon(
            [
                (adata.obsm["X_spatial"][i, 0] - 0.3, adata.obsm["X_spatial"][i, 1] - 0.3),
                (adata.obsm["X_spatial"][i, 0] + 0.3, adata.obsm["X_spatial"][i, 1] - 0.3),
                (adata.obsm["X_spatial"][i, 0] + 0.3, adata.obsm["X_spatial"][i, 1] + 0.3),
                (adata.obsm["X_spatial"][i, 0] - 0.3, adata.obsm["X_spatial"][i, 1] + 0.3),
            ]
        )
        for i, name in enumerate(adata.obs_names)
    }
    coords = adata.obsm["X_spatial"]
    xr = coords[:, 0].max() - coords[:, 0].min()
    yr = coords[:, 1].max() - coords[:, 1].min()

    fig, ax = plot_segmentation(
        adata,
        color_by="gene1",
        segmentation=seg,
        scalebar=False,
        y_pad_ratio=0.2,
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()  # inverted: ylim[0] > ylim[1]
    np.testing.assert_allclose(xlim[1] - xlim[0], xr, atol=1e-9)
    # padded y range = yr * (1 + 2 * 0.2) = 1.4 * yr
    np.testing.assert_allclose(abs(ylim[0] - ylim[1]), yr * 1.4, atol=1e-9)
    plt.close(fig)


def test_plot_segmentation_legend_params_override():
    adata = create_controlled_adata_with_spatial()
    fig, ax = plot_segmentation(
        adata,
        color_by="group",
        celltype_key="group",
        scalebar=False,
        legend_fontsize=7.0,
        legend_params={"fontsize": 14, "labelspacing": 2.0},
    )
    leg = ax.get_legend()
    # legend_params overrides the dedicated arg
    assert leg.get_texts()[0].get_fontsize() == 14
    plt.close(fig)


def test_plot_segmentation_x_pad_ratio():
    adata = create_controlled_adata_with_spatial()
    coords = adata.obsm["X_spatial"]
    xr = coords[:, 0].max() - coords[:, 0].min()

    fig, ax = plot_segmentation(
        adata,
        color_by="group",
        celltype_key="group",
        scalebar=False,
        x_pad_ratio=0.25,
    )
    xlim = ax.get_xlim()
    np.testing.assert_allclose(xlim[1] - xlim[0], xr * 1.5, atol=1e-9)
    plt.close(fig)


def test_plot_segmentation_multi_panel_returns_axes_array():
    adata = create_controlled_adata_with_spatial()
    fig, axes = plot_segmentation(
        adata,
        color_by=["gene1", "gene2"],
        scalebar=False,
        ncols=2,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (1, 2)
    visible = [a for a in axes.ravel() if a.get_visible()]
    assert len(visible) == 2
    titles = [a.get_title() for a in axes.ravel()]
    assert titles == ["gene1", "gene2"]
    plt.close(fig)


def test_plot_segmentation_multi_panel_ncols_layout():
    """ncols=2 with 5 features => 2 cols x 3 rows, with 1 hidden axis."""
    adata = create_controlled_adata_with_spatial()
    features = ["gene1", "gene2", "gene1", "gene2", "gene1"]
    fig, axes = plot_segmentation(
        adata,
        color_by=features,
        scalebar=False,
        ncols=2,
    )
    assert axes.shape == (3, 2)
    flat = axes.ravel()
    visible = [a.get_visible() for a in flat]
    assert visible == [True, True, True, True, True, False]
    plt.close(fig)


def test_plot_segmentation_multi_panel_figsize_per_panel():
    """In multi-panel mode `figsize` is the per-panel size."""
    adata = create_controlled_adata_with_spatial()
    fig, axes = plot_segmentation(
        adata,
        color_by=["gene1", "gene2", "gene1", "gene2"],
        scalebar=False,
        ncols=2,
        figsize=(3, 3),
    )
    w, h = fig.get_size_inches()
    assert pytest.approx(w, rel=1e-9) == 3 * 2  # ncols=2
    assert pytest.approx(h, rel=1e-9) == 3 * 2  # nrows=2
    plt.close(fig)


def test_plot_segmentation_multi_panel_with_ax_raises():
    adata = create_controlled_adata_with_spatial()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="single `color_by`"):
        plot_segmentation(
            adata,
            color_by=["gene1", "gene2"],
            ax=ax,
            scalebar=False,
        )
    plt.close(fig)


def test_plot_segmentation_multi_panel_mixed_categorical_continuous():
    """A list mixing an obs categorical and a gene continuous should both render."""
    adata = create_controlled_adata_with_spatial()
    fig, axes = plot_segmentation(
        adata,
        color_by=["group", "gene1"],
        celltype_key="group",
        scalebar=False,
        ncols=2,
    )
    # Categorical panel has a legend; continuous panel has a colorbar (extra fig axis).
    cat_ax = axes[0, 0]
    assert cat_ax.get_legend() is not None
    # continuous colorbar adds an extra axes to the figure beyond the 2 panel axes
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_plot_segmentation_multi_panel_panel_titles_off():
    adata = create_controlled_adata_with_spatial()
    fig, axes = plot_segmentation(
        adata,
        color_by=["gene1", "gene2"],
        scalebar=False,
        ncols=2,
        panel_titles=False,
    )
    titles = [a.get_title() for a in axes.ravel()]
    assert titles == ["", ""]
    plt.close(fig)


def test_plot_segmentation_single_feature_default_figsize_unchanged():
    """Backwards compat: single-feature default figsize stays at (10, 10)."""
    adata = create_controlled_adata_with_spatial()
    fig, ax = plot_segmentation(
        adata, color_by="group", celltype_key="group", scalebar=False,
    )
    w, h = fig.get_size_inches()
    assert (w, h) == (10.0, 10.0)
    plt.close(fig)
