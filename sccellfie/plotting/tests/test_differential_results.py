import tempfile
import os
import pathlib
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sccellfie.plotting.differential_results import create_volcano_plot, create_comparative_violin, create_beeswarm_plot
from sccellfie.datasets.toy_inputs import create_controlled_adata


def test_create_volcano_plot():
    # Create a sample DataFrame with the new format
    de_results = pd.DataFrame({
        'cell_type': ['TypeA'] * 5 + ['TypeB'] * 5,
        'feature': ['gene1', 'gene2', 'gene3', 'gene4', 'gene5'] * 2,
        'group1': ['A'] * 10,
        'group2': ['B'] * 10,
        'log2FC': [1.5, -0.5, 3.0, -2.0, 0.1] * 2,
        'test_statistic': [2.5, -1.0, 4.0, -3.5, 0.2] * 2,
        'p_value': [0.01, 0.1, 0.001, 0.005, 0.5] * 2,
        'cohens_d': [0.8, -0.3, 1.5, -1.2, 0.05] * 2,
        'adj_p_value': [0.02, 0.15, 0.003, 0.01, 0.6] * 2,
        'n_group1': [100] * 10,
        'n_group2': [100] * 10,
        'median_group1': [1.0] * 10,
        'median_group2': [2.0] * 10,
        'median_diff': [1.0] * 10
    })

    # Test with default parameters
    significant_features = create_volcano_plot(de_results)
    assert isinstance(significant_features, list)
    assert len(significant_features) == 3  # genes 1, 3, and 4 should be significant
    assert set(significant_features) == {'gene1', 'gene3', 'gene4'}

    # Test with custom thresholds
    significant_features = create_volcano_plot(de_results, effect_threshold=1.0, padj_threshold=0.01)
    assert len(significant_features) == 1  # only gene 3 should be significant
    assert significant_features == ['gene3']

    # Test with specific cell type and groups
    significant_features = create_volcano_plot(de_results, cell_type='TypeA', group1='A', group2='B')
    assert len(significant_features) == 3  # genes 1, 3, and 4 should be significant for TypeA
    assert set(significant_features) == {'gene1', 'gene3', 'gene4'}

    # Test with log2FC as effect size
    significant_features = create_volcano_plot(de_results, effect_col='log2FC', effect_title='log2 Fold Change')
    assert len(significant_features) == 3  # genes 1, 3, and 4 should be significant
    assert set(significant_features) == {'gene1', 'gene3', 'gene4'}

    # Test saving the plot
    import tempfile
    import os
    import pathlib
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create pathlib Path object for scanpy settings
        tmp_path = pathlib.Path(tmpdirname)

        # Store original figdir
        original_figdir = sc.settings.figdir

        # Set scanpy figdir to our temporary directory
        sc.settings.figdir = tmp_path
        os.makedirs(tmp_path, exist_ok=True)

        try:
            # Test with relative path
            save_path = 'test_plot.png'
            create_volcano_plot(de_results, save=save_path)
            expected_path = os.path.join(tmpdirname, 'volcano_test_plot.png')
            assert os.path.exists(expected_path), f"File not found at {expected_path}"

            # Test with absolute path and filters
            abs_save_path = os.path.join(tmpdirname, 'abs_test.png')
            create_volcano_plot(
                de_results,
                cell_type='TypeA',
                group1='A',
                group2='B',
                save=abs_save_path
            )
            expected_abs_path = os.path.join(tmpdirname, 'volcano_abs_test_TypeA_A_vs_B.png')
            assert os.path.exists(expected_abs_path), f"File not found at {expected_abs_path}"

        finally:
            # Restore original figdir
            sc.settings.figdir = original_figdir

    # Clean up the plot
    plt.close()


def test_create_comparative_violin():
    # Create controlled AnnData object
    adata = create_controlled_adata()

    # Add cell types
    adata.obs['cell_type'] = ['A', 'A', 'A', 'A']

    # Create mock significant features (now just feature names)
    features = ['gene1', 'gene2']

    # Test saving the plot
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'test_plot.png')

        fig, ax = create_comparative_violin(
            adata=adata,
            significant_features=features,  # Now just passing feature names
            group1='A',
            group2='B',
            condition_key='group',
            celltype='A',
            cell_type_key='cell_type',
            xlabel='Genes',
            ylabel='Expression',
            title='Test Plot',
            figsize=(10, 5),
            fontsize=8,
            palette=['red', 'blue'],
            save=save_path,
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
        assert os.path.exists(os.path.join(tmpdirname, 'violin_test_plot.png'))

        # Clean up
        plt.close(fig)


def test_create_beeswarm_plot():
    # Create a sample DataFrame similar to the one used in volcano plot tests
    de_results = pd.DataFrame({
        'cell_type': ['TypeA'] * 5 + ['TypeB'] * 5,
        'feature': ['gene1', 'gene2', 'gene3', 'gene4', 'gene5'] * 2,
        'group1': ['A'] * 10,
        'group2': ['B'] * 10,
        'log2FC': [1.5, -0.5, 3.0, -2.0, 0.1] * 2,
        'cohens_d': [0.8, -0.3, 1.5, -1.2, 0.05] * 2,
        'adj_p_value': [0.02, 0.15, 0.003, 0.01, 0.6] * 2,
    })

    # Test with default parameters
    fig, ax, sig_df = create_beeswarm_plot(de_results)

    # Check return types
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(sig_df, pd.DataFrame)

    # Check significant features DataFrame
    assert len(sig_df) == 6  # 3 significant features * 2 cell types
    assert set(sig_df.index.get_level_values('feature')) == {'gene1', 'gene3', 'gene4'}

    # Test with custom thresholds
    fig, ax, sig_df = create_beeswarm_plot(
        de_results,
        cohen_threshold=1.0,
        pval_threshold=0.01
    )
    assert len(sig_df) == 2  # 1 significant feature * 2 cell types (gene3 in both TypeA and TypeB)

    # Test saving the plot
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create pathlib Path object for scanpy settings
        tmp_path = pathlib.Path(tmpdirname)

        # Store original figdir
        original_figdir = sc.settings.figdir

        # Set scanpy figdir to our temporary directory
        sc.settings.figdir = tmp_path
        os.makedirs(tmp_path, exist_ok=True)

        try:
            # Test with relative path
            save_path = 'test_plot.png'
            create_beeswarm_plot(de_results, save=save_path)
            expected_path = os.path.join(tmpdirname, 'beeswarm_test_plot.png')
            assert os.path.exists(expected_path), f"File not found at {expected_path}"

            # Test with absolute path
            abs_save_path = os.path.join(tmpdirname, 'abs_test.png')
            create_beeswarm_plot(
                de_results,
                save=abs_save_path,
                title='Custom Title'
            )
            expected_abs_path = os.path.join(tmpdirname, 'beeswarm_abs_test.png')
            assert os.path.exists(expected_abs_path), f"File not found at {expected_abs_path}"

        finally:
            # Restore original figdir
            sc.settings.figdir = original_figdir

    # Clean up plots
    plt.close('all')