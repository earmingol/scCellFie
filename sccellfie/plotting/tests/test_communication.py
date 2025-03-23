import pytest
import tempfile
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


from sccellfie.plotting.communication import normalize_vector, orthogonal_vector, draw_self_loop, plot_communication_network


def test_normalize_vector():
    # Test basic normalization
    vector = np.array([3.0, 4.0])  # norm is 5
    normalize_to = 10
    normalized = normalize_vector(vector, normalize_to)
    assert np.allclose(np.linalg.norm(normalized), normalize_to)

    # Test with unit vector
    vector = np.array([1.0, 0.0])
    normalize_to = 5
    normalized = normalize_vector(vector, normalize_to)
    assert np.allclose(normalized, np.array([5.0, 0.0]))


def test_orthogonal_vector():
    # Test basic orthogonal vector
    point = np.array([1.0, 1.0])
    width = 1.0
    ort_vec = orthogonal_vector(point, width)
    # Test orthogonality using dot product with more lenient tolerance
    assert np.abs(np.dot(point, ort_vec)) < 1e-6

    # Test with normalization
    normalize_to = 2.0
    ort_vec_norm = orthogonal_vector(point, width, normalize_to=normalize_to)
    assert np.allclose(np.linalg.norm(ort_vec_norm), normalize_to)

    # Test with near-zero y-component
    point = np.array([1.0, 1e-10])
    ort_vec = orthogonal_vector(point, width)
    assert not np.any(np.isnan(ort_vec))

    # Test with zero y-component
    point = np.array([1.0, 0.0])
    ort_vec = orthogonal_vector(point, width)
    assert not np.any(np.isnan(ort_vec))


def test_draw_self_loop():
    # Create a simple figure and test drawing a self-loop
    fig, ax = plt.subplots()
    point = np.array([0.5, 0.5])
    node_radius = 0.1

    # Set axis limits (needed for draw_self_loop function)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Draw self-loop
    draw_self_loop(point, ax, node_radius)

    # Check that patches were added to the axis
    assert len(ax.patches) > 0

    # Test with different parameters
    draw_self_loop(
        point,
        ax,
        node_radius,
        padding=1.5,
        width=0.2,
        linewidth=2,
        color="blue",
        alpha=0.8,
        mutation_scale=15
    )

    # Clean up
    plt.close(fig)

def test_plot_communication_network():
    # Create sample communication scores DataFrame
    ccc_scores = pd.DataFrame({
        'sender': ['CellA', 'CellB', 'CellA', 'CellC', 'CellA'],
        'receiver': ['CellB', 'CellC', 'CellA', 'CellA', 'CellC'],
        'score': [0.8, 0.6, 0.9, 0.7, 0.5]
    })

    # Test with default parameters
    fig, ax = plot_communication_network(
        ccc_scores,
        sender_col='sender',
        receiver_col='receiver',
        score_col='score'
    )

    # Check return types
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Clean up
    plt.close(fig)

    # Test with score threshold
    fig, ax = plot_communication_network(
        ccc_scores,
        sender_col='sender',
        receiver_col='receiver',
        score_col='score',
        score_threshold=0.7
    )

    # There should be fewer edges due to threshold
    visible_patches = [p for p in ax.patches if p.get_alpha() > 0]
    assert len(visible_patches) == 3  # Only scores >= 0.7 should be visible
    plt.close(fig)

    # Test with different layouts
    # Test circular layout
    fig, ax = plot_communication_network(
        ccc_scores,
        sender_col='sender',
        receiver_col='receiver',
        score_col='score',
        network_layout='circular'
    )
    plt.close(fig)

    # Test invalid layout
    with pytest.raises(ValueError):
        plot_communication_network(
            ccc_scores,
            sender_col='sender',
            receiver_col='receiver',
            score_col='score',
            network_layout='invalid'
        )

    # Test file saving
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_path = os.path.join(tmpdirname, 'test_network.png')
        expected_path = os.path.join(tmpdirname, 'ccc_test_network.png')
        fig, ax = plot_communication_network(
            ccc_scores,
            sender_col='sender',
            receiver_col='receiver',
            score_col='score',
            save=save_path
        )

        assert os.path.exists(expected_path), f"File not found at {expected_path}"
        plt.close(fig)

    # Test with custom aesthetics
    fig, ax = plot_communication_network(
        ccc_scores,
        sender_col='sender',
        receiver_col='receiver',
        score_col='score',
        edge_color='red',
        node_color='blue',
        node_size=2000,
        edge_width=50,
        title='Test Network'
    )

    # Check if title is set
    assert ax.get_title() == 'Test Network'
    plt.close(fig)

    # Test with existing axes
    fig, ax = plt.subplots()
    fig_out, ax_out = plot_communication_network(
        ccc_scores,
        sender_col='sender',
        receiver_col='receiver',
        score_col='score',
        ax=ax
    )

    # Check if the same axes object is returned
    assert ax_out == ax
    plt.close(fig)


def test_self_loops():
    # Create sample data with self-loops
    ccc_scores = pd.DataFrame({
        'sender': ['CellA', 'CellA', 'CellB'],
        'receiver': ['CellA', 'CellB', 'CellB'],
        'score': [0.8, 0.6, 0.7]
    })

    fig, ax = plot_communication_network(
        ccc_scores,
        sender_col='sender',
        receiver_col='receiver',
        score_col='score'
    )

    # There should be visible patches for the self-loops
    visible_patches = [p for p in ax.patches if p.get_alpha() > 0]
    assert len(visible_patches) >= 2  # At least the two self-loops should be present

    plt.close(fig)