import os
import pytest
import scanpy as sc
from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir


@pytest.fixture
def setup_scanpy_settings(tmp_path):
    """Set up temporary scanpy settings for testing."""
    original_figdir = sc.settings.figdir
    sc.settings.figdir = tmp_path
    yield
    sc.settings.figdir = original_figdir


def test_get_file_format_common_formats():
    """Test _get_file_format with common image formats."""
    test_cases = {
        'plot.png': 'png',
        'plot.PDF': 'pdf',
        'plot.svg': 'svg',
        'plot.jpg': 'jpg',
        'plot.jpeg': 'jpeg',
        'plot.tiff': 'tiff'
    }

    for filename, expected in test_cases.items():
        assert _get_file_format(filename) == expected


def test_get_file_format_no_extension():
    """Test _get_file_format with no extension."""
    assert _get_file_format('plot') == sc.settings.file_format_figs


def test_get_file_format_invalid_extension():
    """Test _get_file_format with invalid extension."""
    assert _get_file_format('plot.txt') == sc.settings.file_format_figs


def test_get_file_dir_absolute_path():
    """Test _get_file_dir with absolute path."""
    if os.name == 'nt':  # Windows
        abs_path = 'C:\\Users\\test\\plot.png'
        expected_dir = 'C:\\Users\\test'
    else:  # Unix-like
        abs_path = '/home/user/plot.png'
        expected_dir = '/home/user'

    dir_path, basename = _get_file_dir(abs_path)
    assert dir_path == expected_dir
    assert basename == 'plot'


def test_get_file_dir_relative_path(setup_scanpy_settings):
    """Test _get_file_dir with relative path."""
    relative_path = 'plot.png'
    dir_path, basename = _get_file_dir(relative_path)

    assert dir_path == str(sc.settings.figdir.absolute())
    assert basename == 'plot'


def test_get_file_dir_nested_path():
    """Test _get_file_dir with nested relative path."""
    nested_path = 'subfolder/plot.png'
    dir_path, basename = _get_file_dir(nested_path)

    # For a nested path, expect the absolute path to the subfolder
    expected_path = os.path.abspath('subfolder')
    assert dir_path == expected_path
    assert basename == 'plot'