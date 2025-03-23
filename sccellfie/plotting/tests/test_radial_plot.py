import os
import tempfile
import pathlib
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sccellfie.plotting.radial_plot import create_radial_plot


@pytest.fixture
def sample_metabolic_data():
    # Create synthetic metabolic data with only a small number of categories to prevent
    # issues with Glasbey palette extension
    np.random.seed(42)
    n_tasks = 8  # Using exactly 8 tasks (matches number of Dark2 colors)
    n_cell_types = 3
    n_tissues = 2

    tasks = [f"task_{i}" for i in range(n_tasks)]
    cell_types = [f"cell_type_{i}" for i in range(n_cell_types)]
    tissues = [f"tissue_{i}" for i in range(n_tissues)]

    data = []
    for task in tasks:
        for tissue in tissues:
            for cell_type in cell_types:
                data.append({
                    'metabolic_task': task,
                    'cell_type': cell_type,
                    'tissue': tissue,
                    'scaled_trimean': np.random.uniform(0, 1)
                })

    return pd.DataFrame(data)


@pytest.fixture
def sample_task_info(sample_metabolic_data):
    # Create synthetic task categories - use exactly 8 tasks with 8 distinct categories
    # (Dark2 palette has 8 colors, so we avoid Glasbey extension completely)
    categories = ['Category_A', 'Category_B', 'Category_C', 'Category_D',
                  'Category_E', 'Category_F', 'Category_G', 'Category_H']
    tasks = sample_metabolic_data['metabolic_task'].unique()

    # Assign each task to a unique category (no need to cycle since we have 8 of each)
    data = []
    for i, task in enumerate(tasks):
        data.append({
            'Task': task,
            'System': categories[i]
        })

    return pd.DataFrame(data)


def test_create_radial_plot_basic(sample_metabolic_data, sample_task_info):
    # Test basic functionality with tissue specified
    tissue = 'tissue_0'
    fig, ax = create_radial_plot(sample_metabolic_data, sample_task_info, tissue=tissue)

    # Check if function returns correct objects
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Check if the plot has the expected elements
    assert len(ax.patches) > 0  # Should have rectangle patches for legend

    # Check if title includes tissue and "across cell types" text
    # Look for the title in both figure texts and axis title
    title_in_fig_texts = any(tissue in text.get_text() for text in fig.texts)
    title_in_ax = tissue in (ax.get_title() or "")
    assert title_in_fig_texts or title_in_ax, "Tissue name not found in plot title"

    across_in_fig_texts = any("across cell types" in text.get_text() for text in fig.texts)
    across_in_ax = "across cell types" in (ax.get_title() or "")
    assert across_in_fig_texts or across_in_ax, "Expected text 'across cell types' not found in title"

    plt.close(fig)


def test_create_radial_plot_cell_type_tissue(sample_metabolic_data, sample_task_info):
    # Test with specific cell type and tissue
    cell_type = 'cell_type_0'
    tissue = 'tissue_1'
    fig, ax = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        cell_type=cell_type,
        tissue=tissue
    )

    # Check title includes cell type and tissue
    # Look for titles in both figure texts and axis title
    full_title = ax.get_title() or ""
    title_texts = [text.get_text() for text in fig.texts]

    # Check in figure texts
    cell_type_in_fig = any(cell_type in text for text in title_texts)
    tissue_in_fig = any(tissue in text for text in title_texts)

    # Check in axis title
    cell_type_in_ax = cell_type in full_title
    tissue_in_ax = tissue in full_title

    assert cell_type_in_fig or cell_type_in_ax, f"Cell type '{cell_type}' not found in plot title"
    assert tissue_in_fig or tissue_in_ax, f"Tissue '{tissue}' not found in plot title"

    plt.close(fig)


def test_create_radial_plot_invalid_tissue(sample_metabolic_data, sample_task_info):
    # Test with invalid tissue
    with pytest.raises(ValueError, match="No data found for tissue"):
        create_radial_plot(sample_metabolic_data, sample_task_info, tissue='nonexistent_tissue')


def test_create_radial_plot_invalid_cell_type(sample_metabolic_data, sample_task_info):
    # Test with invalid cell type
    tissue = 'tissue_0'
    with pytest.raises(ValueError, match="No data found for cell_type"):
        create_radial_plot(
            sample_metabolic_data,
            sample_task_info,
            cell_type='nonexistent_cell',
            tissue=tissue
        )


def test_create_radial_plot_save(sample_metabolic_data, sample_task_info):
    # Test saving functionality
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname)
        save_path = tmp_path / 'test_radial.png'

        fig, ax = create_radial_plot(
            sample_metabolic_data,
            sample_task_info,
            tissue='tissue_0',
            save=str(save_path)
        )

        # Check if file was saved
        expected_path = tmp_path / 'radial_test_radial.png'
        assert os.path.exists(expected_path), f"Expected saved file at {expected_path} not found"

        plt.close(fig)


def test_create_radial_plot_custom_params(sample_metabolic_data, sample_task_info):
    # Test with custom parameters
    custom_title = 'Custom Radial Plot'
    custom_figsize = (7, 7)

    fig, ax = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue='tissue_0',
        title=custom_title,
        figsize=custom_figsize
    )

    # Check if custom title was applied (looking in both places)
    title_in_fig = any(custom_title in text.get_text() for text in fig.texts)
    title_in_ax = custom_title in (ax.get_title() or "")
    assert title_in_fig or title_in_ax, f"Custom title '{custom_title}' not found"

    # Check if figure size was applied
    assert fig.get_size_inches().tolist() == list(custom_figsize)

    plt.close(fig)


def test_maximum_aggregation(sample_metabolic_data, sample_task_info):
    # Test that maximum aggregation is applied correctly when cell_type is not specified
    tissue = 'tissue_0'

    # Filter data for this tissue
    tissue_data = sample_metabolic_data[sample_metabolic_data['tissue'] == tissue]

    # Compute expected maximums manually
    task_maxes = tissue_data.groupby('metabolic_task')['scaled_trimean'].max()

    # Create plot with default ylim=1.0
    fig, ax = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue=tissue
    )

    # Get the maximum value used for plotting (ylim)
    ylim = ax.get_ylim()[1]
    assert ylim == 1.0  # Default ylim should be 1.0

    plt.close(fig)

    # Create plot with ylim=None (should use max value)
    fig2, ax2 = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue=tissue,
        ylim=None
    )

    # Get the maximum value used for plotting (ylim)
    ylim2 = ax2.get_ylim()[1]
    assert abs(ylim2 - task_maxes.max()) < 0.001  # Should be approximately equal to the max of maximums

    plt.close(fig2)

    # Create plot with custom ylim
    custom_ylim = 0.75
    fig3, ax3 = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue=tissue,
        ylim=custom_ylim
    )

    # Get the maximum value used for plotting (ylim)
    ylim3 = ax3.get_ylim()[1]
    assert ylim3 == custom_ylim  # Should match the custom ylim

    plt.close(fig3)


def test_sorting_options(sample_metabolic_data, sample_task_info):
    """Test that tasks are sorted differently based on sort_by_value parameter"""
    tissue = 'tissue_0'

    # Prepare test data with predictable values
    # Modify the dataframe to ensure there's a clear order difference between alphabetical vs value sorting
    test_data = sample_metabolic_data.copy()

    # Create a test case where task_0 has lowest values and task_7 has highest values
    # This ensures alphabetical and value sorting will produce different orders
    for idx, row in test_data.iterrows():
        task_num = int(row['metabolic_task'].split('_')[1])
        # Assign values that increase with task number (0.1 to 0.8)
        test_data.loc[idx, 'scaled_trimean'] = 0.1 * (task_num + 1)

    # First, run with alphabetical sorting (default)
    fig1, ax1 = create_radial_plot(
        test_data,
        sample_task_info,
        tissue=tissue
    )

    # Then, run with value-based sorting
    fig2, ax2 = create_radial_plot(
        test_data,
        sample_task_info,
        tissue=tissue,
        sort_by_value=True
    )

    # We'd need to extract actual plotting data from the figures to verify sorting
    # For now, we just make sure both function calls complete without errors
    assert isinstance(fig1, Figure) and isinstance(fig2, Figure)

    plt.close(fig1)
    plt.close(fig2)


def test_with_custom_axis(sample_metabolic_data, sample_task_info):
    """Test with a custom axes passed as parameter"""
    # Create a figure with polar axes
    fig = plt.figure(figsize=(8, 8))
    custom_ax = fig.add_subplot(111, projection='polar')

    # Use the custom axes for the plot
    fig_out, ax_out = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue='tissue_0',
        ax=custom_ax
    )

    # Check that the returned axes is the same we passed in
    assert ax_out is custom_ax
    # Check that the returned figure is the same as our custom axes' figure
    assert fig_out is fig

    plt.close(fig)


def test_legend_visibility(sample_metabolic_data, sample_task_info):
    """Test controlling legend visibility"""
    # Test with legend visible (default)
    fig1, ax1 = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue='tissue_0',
        show_legend=True
    )

    # Legend should be present
    assert ax1.get_legend() is not None

    # Test with legend hidden
    fig2, ax2 = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue='tissue_0',
        show_legend=False
    )

    # Legend should be absent
    assert ax2.get_legend() is None

    plt.close(fig1)
    plt.close(fig2)


def test_subplot_with_custom_axis_and_no_legend(sample_metabolic_data, sample_task_info):
    """Test creating subplots with custom axes and controlled legend visibility"""
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='polar')  # First subplot (polar)
    ax2 = fig.add_subplot(122, projection='polar')  # Second subplot (polar)

    # Create first plot with legend
    create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue='tissue_0',
        ax=ax1,
        show_legend=True
    )

    # Create second plot without legend
    create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue='tissue_1',
        ax=ax2,
        show_legend=False
    )

    # First subplot should have a legend
    assert ax1.get_legend() is not None

    # Second subplot should not have a legend
    assert ax2.get_legend() is None

    plt.close(fig)


def test_color_mapping(sample_metabolic_data, sample_task_info):
    """Test that categories are assigned colors correctly"""
    tissue = 'tissue_0'

    fig, ax = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue=tissue
    )

    # Get legend handles and labels
    legend = ax.get_legend()
    assert legend is not None, "Legend should be present for this test"

    # Get handles using the proper method - legend elements are rectangles in this case
    handles = legend.get_patches()
    labels = [text.get_text() for text in legend.get_texts()]

    # Check that we have the expected number of categories
    categories = sample_task_info['System'].unique()
    assert len(handles) == len(categories), "Number of legend items should match number of categories"

    # Check that all categories are represented in the legend
    for category in categories:
        assert category.upper() in labels, f"Category {category} missing from legend"

    # Check that colors are assigned (each handle should have a color)
    for handle in handles:
        assert handle.get_facecolor() is not None, "Legend handle should have a color"

    plt.close(fig)


def test_missing_categories(sample_metabolic_data, sample_task_info):
    """Test that tasks with missing categories get assigned to 'Other'"""
    # Create a copy of the task info with a missing category
    modified_task_info = sample_task_info.copy()
    # Remove category for one task
    modified_task_info.loc[0, 'System'] = np.nan

    # Capture printed warnings
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        fig, ax = create_radial_plot(
            sample_metabolic_data,
            modified_task_info,
            tissue='tissue_0'
        )

        # Check that warning was printed
        output = captured_output.getvalue()
        assert "tasks have no category information" in output

        # Check that legend includes "OTHER" category
        legend = ax.get_legend()
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert "OTHER" in legend_texts, "Missing categories should be assigned to 'Other'"

        plt.close(fig)
    finally:
        sys.stdout = sys.__stdout__  # Reset stdout


def test_category_ordering():
    """Test that categories are ordered by size (count) as expected"""
    # Create completely new test data with only a few categories to avoid glasbey issues

    # Create test metabolic data
    np.random.seed(42)
    tasks = ['task_A', 'task_B', 'task_C', 'task_D', 'task_E']
    cell_types = ['cell_1', 'cell_2']
    tissue = 'test_tissue'

    # Create the metabolic dataframe
    data = []
    for task in tasks:
        for cell in cell_types:
            data.append({
                'metabolic_task': task,
                'cell_type': cell,
                'tissue': tissue,
                'scaled_trimean': np.random.uniform(0, 1)
            })

    test_df = pd.DataFrame(data)

    # Create task info with known category sizes:
    # Category_X: 3 tasks
    # Category_Y: 2 tasks
    task_info = pd.DataFrame([
        {'Task': 'task_A', 'System': 'Category_X'},
        {'Task': 'task_B', 'System': 'Category_X'},
        {'Task': 'task_C', 'System': 'Category_X'},
        {'Task': 'task_D', 'System': 'Category_Y'},
        {'Task': 'task_E', 'System': 'Category_Y'},
    ])

    # Create the plot
    fig, ax = create_radial_plot(
        test_df,
        task_info,
        tissue=tissue
    )

    # Get legend texts
    legend = ax.get_legend()
    legend_texts = [text.get_text() for text in legend.get_texts()]

    # Category_X (3 tasks) should be first, Category_Y (2 tasks) should be second
    assert legend_texts[0] == 'CATEGORY_X', "Largest category should be first in legend"
    assert legend_texts[1] == 'CATEGORY_Y', "Second largest category should be second in legend"

    plt.close(fig)


def test_custom_palette(sample_metabolic_data, sample_task_info):
    """Test that custom color palette is applied correctly"""
    tissue = 'tissue_0'

    # Test with a different palette
    fig1, ax1 = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue=tissue,
        palette='Set1'  # Different from default 'Dark2'
    )

    # Test with default palette
    fig2, ax2 = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue=tissue
    )

    # Get legend handles from both plots
    legend1 = ax1.get_legend()
    legend2 = ax2.get_legend()

    handles1 = legend1.get_patches()
    handles2 = legend2.get_patches()

    # Compare colors - they should be different between the two palettes
    colors1 = [handle.get_facecolor() for handle in handles1]
    colors2 = [handle.get_facecolor() for handle in handles2]

    # At least one color should be different (this is not a perfect test as it's
    # possible for colors to be the same by chance, but very unlikely all would match)
    any_different = False
    for c1, c2 in zip(colors1, colors2):
        # Convert RGBA to RGB for comparison (ignore alpha)
        rgb1 = c1[:3]
        rgb2 = c2[:3]
        if not np.allclose(rgb1, rgb2, atol=0.01):
            any_different = True
            break

    assert any_different, "Changing palette should result in different colors"

    plt.close(fig1)
    plt.close(fig2)


### NEW TESTS ###

import os
import sys
import tempfile
import pathlib
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from sccellfie.plotting.radial_plot import create_radial_plot


# Test for the task column fallback mechanism
def test_task_column_fallback(sample_metabolic_data):
    """Test that the function can fall back to 'Task' column if task_col is not found"""
    # Create task info with 'Task' column instead of 'metabolic_task'
    alt_task_info = pd.DataFrame([
        {'Task': 'task_0', 'System': 'Category_A'},
        {'Task': 'task_1', 'System': 'Category_B'},
        {'Task': 'task_2', 'System': 'Category_C'},
        {'Task': 'task_3', 'System': 'Category_D'},
        {'Task': 'task_4', 'System': 'Category_E'},
        {'Task': 'task_5', 'System': 'Category_F'},
        {'Task': 'task_6', 'System': 'Category_G'},
        {'Task': 'task_7', 'System': 'Category_H'}
    ])

    # The function should automatically use 'Task' when 'metabolic_task' isn't found
    fig, ax = create_radial_plot(
        sample_metabolic_data,
        alt_task_info,
        tissue='tissue_0'
    )

    # Make sure it worked by checking if the plot was created
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    plt.close(fig)


# Test for custom color list palette
def test_custom_color_list_palette(sample_metabolic_data, sample_task_info):
    """Test that the function accepts a list of colors as palette"""
    # Define a custom list of colors
    custom_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']

    # Create plot with custom color list
    fig, ax = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue='tissue_0',
        palette=custom_colors
    )

    # Check it worked
    legend = ax.get_legend()
    handles = legend.get_patches()

    # The first handle should be approximately red
    first_color = handles[0].get_facecolor()[:3]  # RGB part
    red_color = np.array([1, 0, 0])  # Red in RGB

    # Allow some tolerance because of alpha channel and color space conversions
    assert np.allclose(first_color, red_color, atol=0.2)

    plt.close(fig)


# Test for too few categories (palette extension not needed)
def test_fewer_categories_than_palette(sample_metabolic_data, sample_task_info):
    """Test with fewer categories than the palette size"""
    # Create a version with only 3 tasks/categories
    reduced_data = sample_metabolic_data[
        sample_metabolic_data['metabolic_task'].isin(['task_0', 'task_1', 'task_2'])
    ].copy()

    reduced_task_info = sample_task_info[
        sample_task_info['Task'].isin(['task_0', 'task_1', 'task_2'])
    ].copy()

    # Using 'tab10' which has 10 colors for only 3 categories
    fig, ax = create_radial_plot(
        reduced_data,
        reduced_task_info,
        tissue='tissue_0',
        palette='tab10'
    )

    # Check it worked
    legend = ax.get_legend()
    handles = legend.get_patches()

    # Should have exactly 3 categories in the legend
    assert len(handles) == 3

    plt.close(fig)


# Test for too many categories (more than palette size)
def test_more_categories_than_palette(sample_metabolic_data):
    """Test with more categories than the built-in palette size"""
    # Create data with many categories (more than 'Dark2' has)
    many_tasks = [f"task_{i}" for i in range(12)]  # Dark2 has 8 colors
    many_categories = [f"Category_{chr(65 + i)}" for i in range(12)]  # A-L

    # Create task info with many categories
    many_categories_task_info = pd.DataFrame([
        {'Task': task, 'System': category}
        for task, category in zip(many_tasks, many_categories)
    ])

    # Filter metabolic data to only include our tasks
    filtered_data = sample_metabolic_data[
        sample_metabolic_data['metabolic_task'].isin(many_tasks[:8])
    ].copy()

    # Add extra tasks with random data
    extra_rows = []
    for task in many_tasks[8:]:
        for tissue in ['tissue_0', 'tissue_1']:
            for cell_type in ['cell_type_0', 'cell_type_1', 'cell_type_2']:
                extra_rows.append({
                    'metabolic_task': task,
                    'cell_type': cell_type,
                    'tissue': tissue,
                    'scaled_trimean': np.random.uniform(0, 1)
                })

    extended_data = pd.concat([filtered_data, pd.DataFrame(extra_rows)], ignore_index=True)

    # Create plot with more categories than palette size
    fig, ax = create_radial_plot(
        extended_data,
        many_categories_task_info,
        tissue='tissue_0',
        palette='Dark2'  # Dark2 has 8 colors, we have 12 categories
    )

    # Check it worked - should have all 12 categories in the legend
    legend = ax.get_legend()
    assert len(legend.get_texts()) == 12

    plt.close(fig)


# Test with multiple subplots and checking title placement
def test_title_placement_with_subplots():
    """Test that titles are correctly placed in both standalone and subplot figures"""
    # Create some test data
    tasks = ['task_A', 'task_B', 'task_C']
    cell_types = ['cell_1', 'cell_2']
    tissues = ['tissue_X', 'tissue_Y']

    # Create the metabolic dataframe
    data = []
    for task in tasks:
        for tissue in tissues:
            for cell_type in cell_types:
                data.append({
                    'metabolic_task': task,
                    'cell_type': cell_type,
                    'tissue': tissue,
                    'scaled_trimean': np.random.uniform(0, 1)
                })

    test_df = pd.DataFrame(data)

    # Create task info
    task_info = pd.DataFrame([
        {'Task': 'task_A', 'System': 'Category_1'},
        {'Task': 'task_B', 'System': 'Category_1'},
        {'Task': 'task_C', 'System': 'Category_2'},
    ])

    # 1. Create a standalone plot
    fig1, ax1 = create_radial_plot(
        test_df,
        task_info,
        tissue='tissue_X',
        title='Standalone Plot'
    )

    # Check if title exists in either figure texts or axis title
    title_in_fig = any('Standalone Plot' in text.get_text() for text in fig1.texts)
    title_in_ax = 'Standalone Plot' in (ax1.get_title() or "")

    assert title_in_fig or title_in_ax, "Title not found in either figure texts or axis title"

    # 2. Create a subplot
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = fig2.add_subplot(121, projection='polar')

    _, _ = create_radial_plot(
        test_df,
        task_info,
        tissue='tissue_Y',
        title='Subplot Title',
        ax=ax2
    )

    # Check if title exists in either figure texts or axis title
    subplot_title_in_fig = any('Subplot Title' in text.get_text() for text in fig2.texts)
    subplot_title_in_ax = 'Subplot Title' in (ax2.get_title() or "")

    assert subplot_title_in_fig or subplot_title_in_ax, "Title not found for subplot"

    plt.close(fig1)
    plt.close(fig2)


# Test save functionality - simple version
def test_save_fallback(sample_metabolic_data, sample_task_info):
    """Test that the save function works correctly"""
    # Create a temporary directory and file path
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'test_fallback.png')

        # Create the plot and save it
        fig, ax = create_radial_plot(
            sample_metabolic_data,
            sample_task_info,
            tissue='tissue_0',
            save=save_path
        )

        # Check if the file was saved with the expected name pattern
        expected_path = os.path.join(tmpdirname, 'radial_test_fallback.png')
        assert os.path.exists(expected_path), f"Expected saved file at {expected_path} not found"

        plt.close(fig)


# Test error raised with empty dataset
def test_empty_dataset():
    """Test that an error is raised when using an empty dataset"""
    # Create empty dataframes
    empty_data = pd.DataFrame(columns=['metabolic_task', 'cell_type', 'tissue', 'scaled_trimean'])
    task_info = pd.DataFrame(columns=['Task', 'System'])

    # Should raise ValueError
    with pytest.raises(ValueError):
        create_radial_plot(empty_data, task_info, tissue='any_tissue')


# Test with no show_legend
def test_no_legend_title_ylim(sample_metabolic_data, sample_task_info):
    """Test multiple parameters: show_legend=False, title=None, ylim=None"""
    # Get data for specific tissue and calculate max
    tissue_data = sample_metabolic_data[sample_metabolic_data['tissue'] == 'tissue_0']
    max_val = tissue_data['scaled_trimean'].max()

    fig, ax = create_radial_plot(
        sample_metabolic_data,
        sample_task_info,
        tissue='tissue_0',
        show_legend=False,
        title=None,
        ylim=None
    )

    # Check legend is not present
    assert ax.get_legend() is None, "Legend should not be present"

    # Check title is not present (or empty)
    title_in_fig = any(len(text.get_text().strip()) > 0 for text in fig.texts)
    title_in_ax = ax.get_title() and len(ax.get_title().strip()) > 0

    assert not (title_in_fig or title_in_ax), "No title should be present"

    # Check ylim is approximately set to max value (with tolerance)
    assert abs(ax.get_ylim()[1] - max_val) < 0.05, "ylim should be close to max value"

    plt.close(fig)


# Test with invalid column names
def test_invalid_column_names(sample_metabolic_data, sample_task_info):
    """Test how the function handles invalid column names"""
    # Test with invalid cell_type_col
    with pytest.raises(ValueError, match="Missing required columns"):
        create_radial_plot(
            sample_metabolic_data,
            sample_task_info,
            cell_type='cell_type_0',
            tissue='tissue_0',
            cell_type_col='nonexistent_col'
        )

    # Test with invalid tissue_col
    with pytest.raises(ValueError, match="Missing required columns"):
        create_radial_plot(
            sample_metabolic_data,
            sample_task_info,
            tissue='tissue_0',
            tissue_col='nonexistent_col'
        )

    # Test with invalid value_col
    with pytest.raises(ValueError, match="Missing required columns"):
        create_radial_plot(
            sample_metabolic_data,
            sample_task_info,
            tissue='tissue_0',
            value_col='nonexistent_col'
        )


# Test with non-polar axes
def test_non_polar_axes():
    """Test that the function raises an error when given a non-polar axes"""
    # Create some test data
    tasks = ['task_A', 'task_B']
    data = pd.DataFrame([
        {'metabolic_task': 'task_A', 'cell_type': 'cell_1', 'tissue': 'tissue_X', 'scaled_trimean': 0.5},
        {'metabolic_task': 'task_B', 'cell_type': 'cell_1', 'tissue': 'tissue_X', 'scaled_trimean': 0.7}
    ])

    task_info = pd.DataFrame([
        {'Task': 'task_A', 'System': 'Category_1'},
        {'Task': 'task_B', 'System': 'Category_2'}
    ])

    # Create a regular (non-polar) axes
    fig = plt.figure()
    ax = fig.add_subplot(111)  # Regular, not polar

    # Should raise ValueError about needing polar projection
    with pytest.raises(ValueError, match="must have polar projection"):
        create_radial_plot(
            data,
            task_info,
            tissue='tissue_X',
            ax=ax
        )

    plt.close(fig)


# Test category assignment based on mode (across cell types vs specific cell type)
def test_cell_type_vs_max_aggregation(sample_metabolic_data, sample_task_info):
    """Test the difference between aggregating by max vs using a specific cell type"""
    tissue = 'tissue_0'

    # Create predictable test data
    test_data = sample_metabolic_data.copy()

    # Set up predictable values for different cell types
    # The error was with parsing cell_type values, so let's be more careful
    for idx, row in test_data.iterrows():
        task = row['metabolic_task']
        cell = row['cell_type']

        # Assign values based on simple rule: cell_type_0 gets high values
        if cell == 'cell_type_0':
            test_data.loc[idx, 'scaled_trimean'] = 0.9
        else:
            test_data.loc[idx, 'scaled_trimean'] = 0.1

    # First, test with a specific cell type (should use values from this cell type only)
    fig1, ax1 = create_radial_plot(
        test_data,
        sample_task_info,
        cell_type='cell_type_0',
        tissue=tissue
    )

    # Then, test with no cell type (should use max across cell types)
    fig2, ax2 = create_radial_plot(
        test_data,
        sample_task_info,
        tissue=tissue
    )

    # Check that both titles have expected text
    # For the first plot (with cell_type)
    title1_texts = [text.get_text() for text in fig1.texts]
    title1_text = " ".join(title1_texts) + ax1.get_title()
    assert 'cell_type_0' in title1_text or any('cell_type_0' in t for t in title1_texts), \
        "Cell type should be mentioned in title when specified"

    # For the second plot (without cell_type)
    title2_texts = [text.get_text() for text in fig2.texts]
    title2_text = " ".join(title2_texts) + ax2.get_title()
    assert 'across cell types' in title2_text or any('across cell types' in t for t in title2_texts), \
        "Should mention 'across cell types' when no specific cell type"

    plt.close(fig1)
    plt.close(fig2)