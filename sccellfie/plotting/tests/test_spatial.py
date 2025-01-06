import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sccellfie.plotting.spatial import plot_spatial, plot_neighbor_distribution
from sccellfie.datasets.toy_inputs import create_controlled_adata_with_spatial


@pytest.fixture
def spatial_adata():
    adata = create_controlled_adata_with_spatial()
    # Add required spatial metadata
    adata.uns['spatial'] = {
        'default': {
            'images': {
                'hires': np.zeros((100, 100)),  # Dummy image
            },
            'scalefactors': {
                'tissue_hires_scalef': 1.0,
                'spot_diameter_fullres': 10,
            },
        }
    }
    return adata


def test_plot_spatial_basic(spatial_adata):
    genes = ['gene1', 'gene2']
    fig, axes = plot_spatial(spatial_adata, genes)

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (1, 2)

    # Account for both plot axes and colorbar axes
    assert len(fig.axes) == 4  # 2 plot axes + 2 colorbars

    plt.close(fig)


def test_plot_spatial_custom_layout(spatial_adata):
    genes = ['gene1', 'gene2', 'gene3']
    fig, axes = plot_spatial(spatial_adata, genes, ncols=2, figsize=(4, 4))

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2, 2)

    # Account for plot axes and colorbars
    assert len(fig.axes) == 6  # 3 plot axes + 3 colorbars

    plt.close(fig)


def test_plot_spatial_data_representation(spatial_adata):
    genes = ['gene1']
    fig, axes = plot_spatial(spatial_adata, genes)

    scatter = axes[0, 0].collections[0]

    # Get normalized coordinates
    coords = scatter.get_offsets()
    expected_coords = spatial_adata.obsm['X_spatial']

    # Check color values instead of coordinates
    colors = scatter.get_array()
    expected_colors = spatial_adata[:, 'gene1'].X.toarray().flatten()
    assert np.allclose(colors, expected_colors)

    plt.close(fig)


def test_plot_spatial_title_and_labels(spatial_adata):
    genes = ['gene1']
    suptitle = "Test Plot"
    fig, axes = plot_spatial(spatial_adata, genes, suptitle=suptitle)

    assert fig.texts[0].get_text() == suptitle
    assert axes[0, 0].get_title() == 'gene1'
    assert axes[0, 0].get_xlabel() == ''
    assert axes[0, 0].get_ylabel() == ''

    plt.close(fig)


@pytest.fixture
def sample_results():
    return {
        'radii': np.array([1.0, 2.0, 3.0, 4.0]),
        'mean': np.array([2.0, 4.0, 6.0, 8.0]),
        'neighbors': np.random.randint(0, 10, size=(100, 4)),
        'quantiles': {
            0.025: np.array([1.0, 2.0, 3.0, 4.0]),
            0.975: np.array([3.0, 6.0, 9.0, 12.0])
        }
    }


def test_plot_neighbor_distribution_basic(sample_results):
    fig, gs = plot_neighbor_distribution(sample_results)

    assert isinstance(fig, plt.Figure)
    assert isinstance(gs, GridSpec)
    assert gs.get_geometry() == (2, 3)

    plt.close(fig)


def test_plot_neighbor_distribution_layout(sample_results):
    fig, gs = plot_neighbor_distribution(sample_results)

    axes = fig.get_axes()
    assert len(axes) == 4  # 1 main plot + 3 histogram subplots

    main_ax = axes[0]
    assert main_ax.get_xlabel() == 'Radius'
    assert main_ax.get_ylabel() == 'Number of Neighbors'

    plt.close(fig)


def test_plot_neighbor_distribution_data(sample_results):
    fig, gs = plot_neighbor_distribution(sample_results)

    main_ax = fig.get_axes()[0]
    lines = main_ax.get_lines()

    # Check mean line data
    mean_line = lines[0]
    assert np.array_equal(mean_line.get_xdata(), sample_results['radii'])
    assert np.array_equal(mean_line.get_ydata(), sample_results['mean'])

    plt.close(fig)


def test_plot_neighbor_distribution_custom_size(sample_results):
    custom_figsize = (10, 6)
    fig, gs = plot_neighbor_distribution(sample_results, figsize=custom_figsize)

    assert fig.get_size_inches().tolist() == list(custom_figsize)

    plt.close(fig)


def test_plot_neighbor_distribution_save(sample_results, tmp_path):
    save_path = tmp_path / "test_plot.png"
    fig, gs = plot_neighbor_distribution(sample_results, save=str(save_path))

    assert save_path.exists()

    plt.close(fig)