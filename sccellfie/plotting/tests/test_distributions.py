import pytest
import numpy as np
import matplotlib.pyplot as plt

from sccellfie.plotting.distributions import create_multi_violin_plots
from sccellfie.datasets.toy_inputs import create_controlled_adata


@pytest.fixture
def controlled_adata():
    return create_controlled_adata()


def test_create_multi_violin_plots_basic(controlled_adata):
    genes = ['gene1', 'gene2', 'gene3']
    fig, axes = create_multi_violin_plots(controlled_adata, genes, groupby='group')

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (1, 4)  # 1 row, 4 columns (adjusted based on the error)

    # Checking if the correct number of subplots are created
    assert len(axes.flat) == 4  # 3 genes + 1 empty subplot

    plt.close(fig)  # Clean up

def test_create_multi_violin_plots_save(controlled_adata):
    # Test saving the plot
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'test_plot.png')

        genes = ['gene1', 'gene2', 'gene3']
        fig, axes = create_multi_violin_plots(controlled_adata, genes, groupby='group', save=save_path)
        assert os.path.exists(os.path.join(tmpdirname, 'violin_test_plot.png'))

        # Clean up the plot
        plt.close()


def test_create_multi_violin_plots_custom_layout(controlled_adata):
    genes = ['gene1', 'gene2']
    fig, axes = create_multi_violin_plots(controlled_adata, genes, groupby='group', n_cols=1, figsize=(3, 2))

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2, 1)  # 2 rows, 1 column

    # Checking if the correct number of subplots are created
    assert len(axes.flat) == 2

    plt.close(fig)  # Clean up


def test_create_multi_violin_plots_data_representation(controlled_adata):
    genes = ['gene1', 'gene2']
    fig, axes = create_multi_violin_plots(controlled_adata, genes, groupby='group')

    # Checking if violins are plotted (4 collections per gene: 2 for violins, 2 for medians)
    assert len(axes[0, 0].collections) == 4
    assert len(axes[0, 1].collections) == 4

    # Checking if the data is correctly represented
    # For gene1: group A should have values [1, 3], group B should have values [5, 7]
    gene1_data = axes[0, 0].collections[0].get_paths()[0].vertices[:, 1]
    assert np.min(gene1_data) == pytest.approx(1, abs=0.1)
    assert np.max(gene1_data) == pytest.approx(3, abs=0.1)

    gene1_data = axes[0, 0].collections[1].get_paths()[0].vertices[:, 1]
    assert np.min(gene1_data) == pytest.approx(5, abs=0.1)
    assert np.max(gene1_data) == pytest.approx(7, abs=0.1)

    plt.close(fig)  # Clean up