import pytest
import os
import tempfile
import pandas as pd

from sccellfie.io.save_data import save_adata, save_result_summary
from sccellfie.spatial import create_knn_network
from sccellfie.datasets.toy_inputs import create_random_adata, create_random_adata_with_spatial

@pytest.mark.parametrize("spatial_data, test_attr", [(True, False), (False, False), (False, True)]) # Test with and without spatial data
def test_save_adata(spatial_data, test_attr):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create sample AnnData
        if spatial_data:
            adata = create_random_adata_with_spatial(n_obs=20, n_vars=10, spatial_key='X_spatial')
        else:
            adata = create_random_adata(n_obs=20, n_vars=10)

        # Add reactions and metabolic tasks objects
        adata.reactions = create_random_adata_with_spatial(n_obs=20, n_vars=10, spatial_key='X_spatial') if spatial_data else create_random_adata(n_obs=20, n_vars=10)
        adata.metabolic_tasks = create_random_adata_with_spatial(n_obs=20, n_vars=10, spatial_key='X_spatial') if spatial_data else create_random_adata(n_obs=20, n_vars=10)

        # Generate spatial network to test storage of networkx object as pandas dataframe
        if spatial_data:
            create_knn_network(adata, n_neighbors=5, spatial_key='X_spatial', added_key='spatial_network')
            create_knn_network(adata.reactions, n_neighbors=5, spatial_key='X_spatial', added_key='spatial_network')
            create_knn_network(adata.metabolic_tasks, n_neighbors=5, spatial_key='X_spatial', added_key='spatial_network')

        if test_attr:
            delattr(adata, 'reactions')
            delattr(adata, 'metabolic_tasks')

        # Call your save function
        save_adata(adata, tmpdirname, 'test_data')

        # Assertions
        assert os.path.exists(os.path.join(tmpdirname, 'test_data.h5ad'))
        if not test_attr:
            assert os.path.exists(os.path.join(tmpdirname, 'test_data_reactions.h5ad'))
            assert os.path.exists(os.path.join(tmpdirname, 'test_data_metabolic_tasks.h5ad'))


def test_save_result_summary_basic():
    """Test basic functionality of save_result_summary."""
    # Create test data
    agg_values = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])
    melted = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

    results_dict = {
        'agg_values': agg_values,
        'melted': melted
    }

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function
        save_result_summary(results_dict, temp_dir)

        # Check that files were created
        assert os.path.exists(os.path.join(temp_dir, 'Agg_values.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'Melted.csv'))

        # Check file contents
        agg_values_read = pd.read_csv(os.path.join(temp_dir, 'Agg_values.csv'), index_col=0)
        melted_read = pd.read_csv(os.path.join(temp_dir, 'Melted.csv'))

        # Compare DataFrames
        pd.testing.assert_frame_equal(agg_values, agg_values_read)
        pd.testing.assert_frame_equal(melted.reset_index(drop=True), melted_read)


def test_save_result_summary_directory_creation():
    """Test that the output directory is created if it doesn't exist."""
    # Create test data
    df = pd.DataFrame({'A': [1, 2, 3]})
    results_dict = {'test': df}

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Specify a subdirectory that doesn't exist yet
        output_dir = os.path.join(temp_dir, 'new_dir')

        # Call the function
        save_result_summary(results_dict, output_dir)

        # Check that the directory was created
        assert os.path.exists(output_dir)

        # Check that the file was created
        assert os.path.exists(os.path.join(output_dir, 'Test.csv'))


def test_save_result_summary_prefix():
    """Test prefix handling."""
    # Create test data
    df = pd.DataFrame({'A': [1, 2, 3]})
    results_dict = {'test': df}

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with a prefix that doesn't end with a dash
        save_result_summary(results_dict, temp_dir, prefix='myprefix')
        assert os.path.exists(os.path.join(temp_dir, 'myprefix-Test.csv'))

        # Test with a prefix that already ends with a dash
        save_result_summary(results_dict, temp_dir, prefix='otherprefix-')
        assert os.path.exists(os.path.join(temp_dir, 'otherprefix-Test.csv'))

        # Test with an empty prefix
        save_result_summary(results_dict, temp_dir, prefix='')
        assert os.path.exists(os.path.join(temp_dir, 'Test.csv'))


def test_save_result_summary_empty_df():
    """Test handling of empty DataFrames."""
    # Create test data
    empty_df = pd.DataFrame()
    non_empty_df = pd.DataFrame({'A': [1, 2, 3]})

    results_dict = {
        'empty': empty_df,
        'non_empty': non_empty_df
    }

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function
        save_result_summary(results_dict, temp_dir)

        # Check that only the non-empty DataFrame was saved
        assert not os.path.exists(os.path.join(temp_dir, 'Empty.csv'))
        assert os.path.exists(os.path.join(temp_dir, 'Non_empty.csv'))


def test_save_result_summary_non_df():
    """Test handling of non-DataFrame values in the dictionary."""
    # Create test data
    df = pd.DataFrame({'A': [1, 2, 3]})
    non_df = "This is not a DataFrame"

    results_dict = {
        'df': df,
        'non_df': non_df
    }

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function
        save_result_summary(results_dict, temp_dir)

        # Check that only the DataFrame was saved
        assert os.path.exists(os.path.join(temp_dir, 'Df.csv'))
        assert not os.path.exists(os.path.join(temp_dir, 'Non_df.csv'))


def test_save_result_summary_index_saving():
    """Test index saving behavior."""
    # Create test data with named indices
    index_df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
    index_df.index.name = 'idx'

    cell_counts = pd.DataFrame({'cell_type': ['A', 'B'], 'count': [10, 20]})
    melted = pd.DataFrame({'feature': ['f1', 'f2'], 'value': [0.1, 0.2]})

    results_dict = {
        'index_df': index_df,
        'cell_counts': cell_counts,
        'melted': melted
    }

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function
        save_result_summary(results_dict, temp_dir)

        # For index_df, we expect the index to be saved
        index_df_read = pd.read_csv(os.path.join(temp_dir, 'Index_df.csv'), index_col=0)
        pd.testing.assert_frame_equal(index_df, index_df_read)

        # For cell_counts and melted, we expect the index not to be saved
        cell_counts_read = pd.read_csv(os.path.join(temp_dir, 'Cell_counts.csv'))
        pd.testing.assert_frame_equal(cell_counts.reset_index(drop=True), cell_counts_read)

        melted_read = pd.read_csv(os.path.join(temp_dir, 'Melted.csv'))
        pd.testing.assert_frame_equal(melted.reset_index(drop=True), melted_read)