import os
import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
import matplotlib.patches as patches


def normalize_vector(vector, normalize_to):
    """
    Makes `vector` norm equal to `normalize_to`

    Parameters
    ----------
    vector : np.array
        Vector to normalize.

    normalize_to : float
        Value to normalize the vector to.

    Returns
    -------
    normalized_vector : np.array
        Normalized vector.
    """
    vector_norm = np.linalg.norm(vector)
    normalized_vector = vector * normalize_to / vector_norm
    return normalized_vector


def orthogonal_vector(point, width, normalize_to=None):
    """
    Gets orthogonal vector to a `point`

    Parameters
    ----------
    point : np.array
        Point to get the orthogonal vector.

    width : float
        Width of the orthogonal vector.

    normalize_to : float, optional (default: None)
        Value to normalize the vector to.

    Returns
    -------
    ort_vector : np.array
        Orthogonal vector to the point.
    """
    EPSILON = 0.000001
    x = width
    y = -x * point[0] / (point[1] + EPSILON)
    ort_vector = np.array([x, y])
    if normalize_to is not None:
        ort_vector = normalize_vector(ort_vector, normalize_to)
    return ort_vector


def draw_self_loop(point, ax, node_radius, padding=1.2, width=0.1,
                   linewidth=1, color="pink", alpha=0.5,
                   mutation_scale=20):
    """
    Draws a loop from `point` to itself, starting from node border

    Parameters
    ----------
    point : np.array
        Point to draw the loop.

    ax : matplotlib.axes.Axes
        Axes object where the loop will be drawn.

    node_radius : float
        Radius of the node.

    padding : float, optional (default: 1.2)
        Padding for the loop.

    width : float, optional (default: 0.1)
        Width of the loop.

    linewidth : float, optional (default: 1)
        Width of the loop line.

    color : str, optional (default: "pink")
        Color of the loop.

    alpha : float, optional (default: 0.5)
        Transparency of the loop.

    mutation_scale : float, optional (default: 20)
        Mutation scale of the loop.
    """
    # Get the center of the plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    center = np.array([(xlim[1] + xlim[0]) / 2, (ylim[1] + ylim[0]) / 2])

    # Center the point
    centered_point = point - center

    # Calculate direction vector from center to point
    direction = centered_point / np.linalg.norm(centered_point)

    # Calculate start point at node border
    start_point = point - direction * node_radius

    # Calculate the loop points from the border
    point_with_padding = padding * (start_point - center)
    ort_vector = orthogonal_vector(centered_point, width, normalize_to=width)

    first_anchor = ort_vector + point_with_padding + center
    second_anchor = -ort_vector + point_with_padding + center

    # Calculate end point slightly before the node border to show arrow
    end_point = point - direction * (node_radius * 0.7)

    # Define path
    verts = [start_point, first_anchor, second_anchor, end_point]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]

    path = MplPath(verts, codes)
    patch = patches.FancyArrowPatch(
        path=path,
        facecolor='none',
        lw=linewidth,
        arrowstyle="-|>",
        color=color,
        alpha=alpha,
        mutation_scale=mutation_scale
    )
    ax.add_patch(patch)


def plot_communication_network(ccc_scores, sender_col, receiver_col, score_col,
                               score_threshold=None, panel_size=(12, 8),
                               network_layout='spring', edge_color='magenta',
                               edge_width=25, edge_arrow_size=20, edge_alpha=0.25,
                               node_color="#210070", node_size=1000, node_alpha=0.9,
                               node_label_size=12, node_label_alpha=0.7,
                               node_label_offset=(0.05, -0.2), title=None,
                               title_fontsize=14, ax=None, save=None, dpi=300, tight_layout=True, bbox_inches='tight'):
    """
    Plots a network of cell-cell communication. 
    Edges represent communication scores between cells. These scores could
    be an overall communication score or a specific ligand-receptor pair score.
    
    Parameters
    ----------
    ccc_scores : pandas.DataFrame
        DataFrame containing the cell-cell communication scores. It should contain
        columns for the sender cell, receiver cell, and the communication score.
        
    sender_col : str
        Column name for the sender cell.
        
    receiver_col : str
        Column name for the receiver cell.
        
    score_col : str
        Column name for the communication score.
        
    score_threshold : float, optional (default: None)
        Threshold for the communication score. If provided, only scores above this
        threshold are plotted.
        
    panel_size : tuple, optional (default: (12, 8))
        Size of the plot panel. Only works if `ax` is None.
        
    network_layout : str, optional (default: 'spring')
        Layout of the network graph. Should be either 'spring' or 'circular'.
        
    edge_color : str, optional (default: 'magenta')
        Color of the edges.
        
    edge_width : float, optional (default: 25)
        Width of the edges.
        
    edge_arrow_size : float, optional (default: 20)
        Size of the edge arrows.
        
    edge_alpha : float, optional (default: 0.25)
        Transparency of the edges.
        
    node_color : str, optional (default: '#210070')
        Color of the nodes.
        
    node_size : int, optional (default: 1000)
        Size of the nodes.
        
    node_alpha : float, optional (default: 0.9)
        Transparency of the nodes.
        
    node_label_size : int, optional (default: 12)
        Font size of the node labels.
        
    node_label_alpha : float, optional (default: 0.7)
        Transparency of the node labels.
        
    node_label_offset : tuple, optional (default: (0.05, -0.2))
        Offset of the node labels.

    title : str, optional (default: None)
        Title of the plot.

    title_fontsize : int, optional (default: 14)
        Font size of the title.

    ax : matplotlib.axes.Axes, optional (default: None)
        Axes object where the plot will be drawn. If None, a new figure is created.

    save : str, optional (default: None)
        Filepath to save the plot. If None, the plot is not saved.

    dpi : int, optional (default: 300)
        Resolution of the saved plot.

    tight_layout : bool, optional (default: True)
        Whether to use tight layout for the plot.

    bbox_inches : str, optional (default: 'tight')
        Bounding box in inches. Only used if `save` is provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    # Filter by threshold if specified
    if score_threshold is not None:
        ccc_scores = ccc_scores[ccc_scores[score_col] >= score_threshold].copy()

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=panel_size)
    else:
        fig = plt.gcf()

    # Calculate figure-dependent scaling factor
    fig_width, fig_height = fig.get_size_inches()
    figsize_factor = np.sqrt(fig_width * fig_height) / np.sqrt(12 * 8)

    # Create network graph
    G = nx.DiGraph()
    all_cells = pd.concat([ccc_scores[sender_col], ccc_scores[receiver_col]]).unique()
    G.add_nodes_from(all_cells)

    for _, row in ccc_scores.iterrows():
        G.add_edge(row[sender_col], row[receiver_col], weight=row[score_col])

    # Set layout
    if network_layout == 'spring':
        pos = nx.spring_layout(G, k=1., seed=888)
    elif network_layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        raise ValueError("network_layout should be either 'spring' or 'circular'")

    # Get edge weights and normalize them
    weights = np.array([G.edges[e]['weight'] for e in G.edges()])
    if len(weights) > 0:
        weights_norm = 0.2 + 0.8 * (weights - weights.min()) / (
            weights.max() - weights.min() if weights.max() != weights.min() else 1)
    else:
        weights_norm = []

    # Calculate node radius for offset calculations
    base_node_radius = np.sqrt(node_size / np.pi) / 150
    node_radius = base_node_radius * figsize_factor

    # Separate self-loops from regular edges
    self_loops = [(u, v) for (u, v) in G.edges() if u == v]
    regular_edges = [(u, v) for (u, v) in G.edges() if u != v]

    def get_offset_positions(pos_src, pos_dst, offset):
        """Calculate offset positions for arrow endpoints"""
        direction = pos_dst - pos_src
        length = np.linalg.norm(direction)
        if length == 0:
            return pos_src, pos_dst
        unit_vec = direction / length
        scaled_offset = offset * figsize_factor
        pos_src_offset = pos_src + unit_vec * scaled_offset
        pos_dst_offset = pos_dst - unit_vec * scaled_offset
        return pos_src_offset, pos_dst_offset

    # Draw regular edges
    edge_weights = dict(zip(G.edges(), weights_norm))

    for i, (u, v) in enumerate(regular_edges):
        pos_u, pos_v = np.array(pos[u]), np.array(pos[v])
        offset = node_radius * 0.8
        pos_src_offset, pos_dst_offset = get_offset_positions(pos_u, pos_v, offset)

        arrow = FancyArrowPatch(
            posA=pos_src_offset,
            posB=pos_dst_offset,
            arrowstyle='-|>',
            connectionstyle=f"arc3,rad=-0.2",
            color=edge_color,
            alpha=edge_alpha,
            linewidth=edge_width * edge_weights[(u, v)],
            mutation_scale=edge_arrow_size * figsize_factor
        )
        ax.add_patch(arrow)

    # Draw self-loops using the improved function
    for u, v in self_loops:
        pos_u = np.array(pos[u])
        draw_self_loop(
            point=pos_u,
            ax=ax,
            node_radius=node_radius,
            padding=1.2,
            width=0.1,
            linewidth=edge_width * edge_weights[(u, v)],
            color=edge_color,
            alpha=edge_alpha,
            mutation_scale=edge_arrow_size
        )

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_color,
                           node_size=node_size,
                           alpha=node_alpha,
                           ax=ax)

    # Add labels
    label_options = {"ec": "k", "fc": "white", "alpha": node_label_alpha}
    label_pos = {k: v + np.array(node_label_offset) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos,
                            font_size=node_label_size,
                            bbox=label_options,
                            ax=ax)

    # Adjust layout
    ax.set_frame_on(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeff = 1.4
    ax.set_xlim((xlim[0] * coeff, xlim[1] * coeff))
    ax.set_ylim((ylim[0] * coeff, ylim[1] * coeff))

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, y=0.9)

    if fig is not None:
        if tight_layout:
            plt.tight_layout()
        if save is not None:
            from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir
            dir, basename = _get_file_dir(save)
            os.makedirs(dir, exist_ok=True)
            format = _get_file_format(save)
            plt.savefig(f'{dir}/ccc_{basename}.{format}', dpi=dpi, bbox_inches=bbox_inches)

    return fig, ax