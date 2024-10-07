import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sccellfie.plotting.differential_results import create_volcano_plot, create_comparative_violin
from sccellfie.tests.toy_inputs import create_controlled_adata

def test_create_volcano_plot():
    # Create a sample DataFrame similar to the output of scanpy_differential_analysis
    de_results = pd.DataFrame({
        'cell_type': ['TypeA'] * 5 + ['TypeB'] * 5,
        'feature': ['gene1', 'gene2', 'gene3', 'gene4', 'gene5'] * 2,
        'contrast': ['A vs B'] * 10,
        'log2FC': [1.5, -0.5, 3.0, -2.0, 0.1] * 2,
        'test_statistic': [2.5, -1.0, 4.0, -3.5, 0.2] * 2,
        'p_value': [0.01, 0.1, 0.001, 0.005, 0.5] * 2,
        'cohens_d': [0.8, -0.3, 1.5, -1.2, 0.05] * 2,
        'adj_p_value': [0.02, 0.15, 0.003, 0.01, 0.6] * 2
    }).set_index(['cell_type', 'feature'])

    # Test with default parameters
    significant_points = create_volcano_plot(de_results)
    assert len(significant_points) == 6  # genes 1, 3, and 4 should be significant for both cell types
    assert set(significant_points) == {('TypeA', 'gene1'), ('TypeA', 'gene3'), ('TypeA', 'gene4'),
                                       ('TypeB', 'gene1'), ('TypeB', 'gene3'), ('TypeB', 'gene4')}

    # Test with custom thresholds
    significant_points = create_volcano_plot(de_results, effect_threshold=1.0, padj_threshold=0.01)
    assert len(significant_points) == 2  # only gene 3 should be significant for both cell types
    assert set(significant_points) == {('TypeA', 'gene3'), ('TypeB', 'gene3')}

    # Test with a specific contrast and cell type
    de_results['contrast'] = ['A vs B', 'B vs C', 'A vs B', 'B vs C', 'A vs B'] * 2
    de_results_ct_A = de_results[de_results.index.get_level_values('cell_type') == 'TypeA']
    significant_points = create_volcano_plot(de_results_ct_A, contrast='B vs C')
    assert len(significant_points) == 1  # only gene 4 should be significant for this contrast and cell type
    assert significant_points[0] == ('TypeA', 'gene4')

    # Test with log2FC as effect size
    significant_points = create_volcano_plot(de_results, effect_col='log2FC', effect_title='log2 Fold Change')
    assert len(significant_points) == 6  # genes 1, 3, and 4 should be significant for both cell types
    assert set(significant_points) == {('TypeA', 'gene1'), ('TypeA', 'gene3'), ('TypeA', 'gene4'),
                                       ('TypeB', 'gene1'), ('TypeB', 'gene3'), ('TypeB', 'gene4')}

    # Test saving the plot
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'volcano_plot.png')
        create_volcano_plot(de_results, save=save_path)
        assert os.path.exists(save_path)

    # Clean up the plot
    plt.close()


def test_compare_adata_features(tmp_path):
    # Create controlled AnnData object
    adata = create_controlled_adata()

    # Add cell types
    adata.obs['cell_type'] = ['A', 'A', 'A', 'A']
    # Create mock significant features
    significant_features = [('A', 'gene1'), ('A', 'gene2')]

    # Call the function
    fig, ax = create_comparative_violin(adata=adata,
                                        significant_features=significant_features,
                                        group1='A',
                                        group2='B',
                                        condition_key='group',
                                        celltype='A',
                                        cell_type_key='cell_type',  # We're using 'group' as both condition and cell type for this test
                                        xlabel='Genes',
                                        ylabel='Expression',
                                        title='Test Plot',
                                        figsize=(10, 5),
                                        fontsize=8,
                                        palette=['red', 'blue'],
                                        filename=str(tmp_path / "test_plot.png"),
                                        dpi=100
                                        )

    # Check that the function returns the expected objects
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Check that the plot has the correct labels and title
    assert ax.get_xlabel() == 'Genes'
    assert ax.get_ylabel() == 'Expression'
    assert ax.get_title() == 'Test Plot'

    # Check that the correct number of violin plots are created
    assert len(ax.collections) == 4  # 2 features * 2 groups

    # Check that the legend is created
    assert ax.get_legend() is not None

    # Check that the file was saved
    assert (tmp_path / "test_plot.png").exists()

    # Clean up
    plt.close(fig)