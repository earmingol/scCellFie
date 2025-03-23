import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_radial_plot(metabolic_df, task_info_df, cell_type=None, tissue=None,
                       task_col='metabolic_task', category_col='System', value_col='scaled_trimean',
                       tissue_col='tissue', cell_type_col='cell_type', figsize=(6, 6),
                       title='Metabolic activities', palette='Dark2', title_fontsize=24,
                       legend_fontsize=14, legend_loc="center left",
                       legend_bbox_to_anchor=(1.1, 0.5), alpha_fill=0.25, alpha_bg=0.1, ylim=1.0,
                       sort_by_value=False, ax=None, show_legend=True, save=None, dpi=300,
                       bbox_inches='tight', tight_layout=True):
    """
    Creates a radial plot of metabolic task activities grouped by category.

    Parameters
    ----------
    metabolic_df : pandas.DataFrame
        DataFrame containing metabolic task activities. Typically, it corresponds
        to the 'melted' dataframe in the outputs from `sccellfie.reports.summary.generate_report_from_adata()`.
        Required columns: task_col, value_col, cell_type_col, tissue_col.

    task_info_df : pandas.DataFrame
        DataFrame containing task categorization information.
        Required columns: task_col and category_col.

    cell_type : str, optional (default: None)
        The specific cell type to plot. If None, the maximum activity across all cell types
        within the specified tissue is used.

    tissue : str, optional (default: None)
        The specific tissue to plot. If None, all tissues are included.

    task_col : str, optional (default: 'metabolic_task')
        The column name in metabolic_df containing task identifiers.

    category_col : str, optional (default: 'System')
        The column name in task_info_df containing category information.

    value_col : str, optional (default: 'scaled_trimean')
        The column name in metabolic_df containing activity values.

    tissue_col : str, optional (default: 'tissue')
        The column name in metabolic_df containing tissue information.

    cell_type_col : str, optional (default: 'cell_type')
        The column name in metabolic_df containing cell type information.

    figsize : tuple, optional (default: (6, 6))
        The size of the figure. Only used if ax is None.

    title : str, optional (default: 'Metabolic activities')
        The title for the plot. Set to None to disable the title.

    palette : str, optional (default: 'Dark2)
        Name of a palette for coloring the categories of metabolic tasks.

    title_fontsize : int, optional (default: 24)
        Font size for the title.

    legend_fontsize : int, optional (default: 14)
        Font size for the legend.

    legend_loc : str, optional (default: "center left")
        Location of the legend.

    legend_bbox_to_anchor : tuple, optional (default: (1.1, 0.5))
        Position of the legend relative to the legend_loc.

    alpha_fill : float, optional (default: 0.25)
        Alpha transparency for the filled areas.

    alpha_bg : float, optional (default: 0.1)
        Alpha transparency for the background areas.

    ylim : float, optional (default: 1.0)
        Limit value for the y-axis (radial direction). If None, the maximum value across
        all tasks is used instead.

    sort_by_value : bool, optional (default: False)
        If True, tasks within each category are sorted by their value.
        If False, tasks are sorted alphabetically within each category.

    ax : matplotlib.axes.Axes, optional (default: None)
        A matplotlib axes with polar projection to draw the plot on.
        If None, a new figure and axes are created.

    show_legend : bool, optional (default: True)
        Whether to display the legend.

    save : str, optional (default: None)
        The filepath to save the figure. If None, the figure is not saved.

    dpi : int, optional (default: 300)
        The resolution of the saved figure.

    bbox_inches : str, optional (default: 'tight')
        The bbox_inches parameter for saving the figure.

    tight_layout : bool, optional (default: True)
        Whether to use tight layout for the plot. Only applied if ax is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    ax : matplotlib.axes.Axes
        The matplotlib axes object.

    Examples
    --------
    >>> import pandas as pd
    >>> from sccellfie.plotting import create_radial_plot
    >>>
    >>> # Load example data
    >>> metabolic_df = pd.read_csv('Melted.csv')
    >>> task_info_df = pd.read_csv('TaskInfo.csv')
    >>>
    >>> # Create radial plot for maximum activities across all cell types in a tissue
    >>> fig, ax = create_radial_plot(metabolic_df, task_info_df, tissue='Blood')
    >>> plt.show()
    >>>
    >>> # Create radial plot for a specific cell type in a specific tissue
    >>> fig, ax = create_radial_plot(metabolic_df, task_info_df, cell_type='T cell', tissue='Blood')
    >>> plt.show()
    >>>
    >>> # Create multiple subplots with shared legend
    >>> fig = plt.figure(figsize=(20, 10))
    >>> ax1 = fig.add_subplot(121, projection='polar')
    >>> ax2 = fig.add_subplot(122, projection='polar')
    >>>
    >>> # First subplot with legend
    >>> create_radial_plot(metabolic_df, task_info_df, tissue='Blood', ax=ax1, show_legend=True)
    >>> # Second subplot without legend
    >>> create_radial_plot(metabolic_df, task_info_df, tissue='Liver', ax=ax2, show_legend=False)
    >>> plt.tight_layout()
    >>> plt.show()
    """
    # Copy dataframes to avoid modifying originals
    metabolic_df = metabolic_df.copy()
    task_info_df = task_info_df.copy()

    # Check required columns
    required_cols = [task_col, value_col, cell_type_col]
    if tissue is not None:
        required_cols.append(tissue_col)

    missing_cols = [col for col in required_cols if col not in metabolic_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metabolic_df: {', '.join(missing_cols)}")

    # Filter by tissue if specified
    if tissue is not None:
        if tissue_col not in metabolic_df.columns:
            raise ValueError(f"'{tissue_col}' column is missing from metabolic_df")

        data = metabolic_df[metabolic_df[tissue_col] == tissue].copy()
        if len(data) == 0:
            raise ValueError(f"No data found for tissue '{tissue}'")
    else:
        data = metabolic_df.copy()

    # Filter by cell type or get max across cell types
    if cell_type is not None:
        if cell_type_col not in data.columns:
            raise ValueError(f"'{cell_type_col}' column is missing from metabolic_df")

        filtered_data = data[data[cell_type_col] == cell_type].copy()
        if len(filtered_data) == 0:
            raise ValueError(f"No data found for cell_type '{cell_type}'")

        # Group by task to get average values (in case there are multiple entries per task for this cell type)
        radial_df = filtered_data.groupby(task_col)[value_col].mean().reset_index()
    else:
        # Group by task and calculate maximum across all cell types
        radial_df = data.groupby(task_col)[value_col].max().reset_index()

    # Ensure task_info_df has columns we need
    if task_col not in task_info_df.columns:
        # Try to find a different column with task names
        if 'Task' in task_info_df.columns:
            task_info_df = task_info_df.rename(columns={'Task': task_col})
        else:
            raise ValueError(f"'{task_col}' column not found in task_info_df")

    # Merge with task categories
    radial_df = pd.merge(radial_df, task_info_df[[task_col, category_col]], on=task_col, how='left')

    # Check for tasks without category information
    missing_categories = radial_df[radial_df[category_col].isna()][task_col].unique()
    if len(missing_categories) > 0:
        # Assign uncategorized tasks to "Other" category
        radial_df.loc[radial_df[category_col].isna(), category_col] = "Other"
        print(f"Warning: {len(missing_categories)} tasks have no category information and were assigned to 'Other'")

    # Get unique categories ordered by count (descending)
    category_counts = radial_df[category_col].value_counts()
    categories_by_size = category_counts.index.tolist()  # Categories ordered by descending count

    # Simplified color assignment approach
    import matplotlib.cm as cm
    from matplotlib.colors import to_rgba

    # Check if we need to extend the palette
    n_categories = len(categories_by_size)

    if isinstance(palette, str):
        # Get the built-in colormap
        cmap = cm.get_cmap(palette)

        # Check if the colormap has enough colors
        if cmap.N >= n_categories:
            # Colormap has enough colors, use it directly
            color_map = {cat: to_rgba(cmap(i / cmap.N)) for i, cat in enumerate(categories_by_size)}
        else:
            # Need more colors, try to extend with glasbey if available
            try:
                import glasbey
                palette_size = n_categories
                try:
                    extended_palette = glasbey.extend_palette(palette, palette_size=palette_size)
                    color_map = {cat: extended_palette[i] for i, cat in enumerate(categories_by_size)}
                except Exception as e:
                    # Glasbey extension failed, fall back to cycling the original colormap
                    print(f"Warning: Could not extend palette: {str(e)}. Using color cycling instead.")
                    color_map = {cat: to_rgba(cmap(i % cmap.N)) for i, cat in enumerate(categories_by_size)}
            except ImportError:
                # Glasbey not available, use cycling of the original colormap
                print("Warning: glasbey module not available. Using color cycling instead.")
                color_map = {cat: to_rgba(cmap(i % cmap.N)) for i, cat in enumerate(categories_by_size)}
    else:
        # If palette is already a list of colors, use it with cycling if needed
        color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories_by_size)}

    # Create figure if ax is not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
    else:
        # Make sure the provided axis has polar projection
        from matplotlib.projections.polar import PolarAxes
        if not isinstance(ax, PolarAxes):
            raise ValueError("The provided ax must have polar projection")
        fig = ax.figure

    if ylim is None:
        ylim = radial_df[value_col].max()

    # Create a DataFrame with pathway and category information
    pathway_data = radial_df.copy()

    # Sort by category count (descending)
    pathway_data['category_count'] = pathway_data[category_col].map(category_counts)

    # Get categories in the order determined by our sorting (by size)
    categories_ordered = pathway_data.sort_values('category_count', ascending=False)[category_col].unique()

    # Drop the temporary column
    pathway_data = pathway_data.drop(columns=['category_count'])

    # Calculate angle for each pathway
    total_pathways = len(pathway_data)
    angle_per_pathway = (2 * np.pi) / total_pathways

    # Initialize for angle assignment
    pathway_data_grouped = []
    current_angle = 0

    # Process each category in our determined order
    for category in categories_ordered:
        # Record the starting angle for this category
        category_start_angle = current_angle

        # Get data for this category
        category_data = pathway_data[pathway_data[category_col] == category].copy()

        # Sort within category: alphabetically by default, by value if requested
        if sort_by_value:
            category_data = category_data.sort_values(value_col, ascending=False)
        else:
            # Sort alphabetically by task name
            category_data = category_data.sort_values(task_col)

        # Create a range of angles for this category
        num_pathways = len(category_data)
        category_angles = np.linspace(current_angle,
                                      current_angle + (num_pathways * angle_per_pathway),
                                      num_pathways,
                                      endpoint=False)

        # Assign these angles
        category_data['angle'] = category_angles

        # Update current angle for the next category
        current_angle += num_pathways * angle_per_pathway

        # Store category boundary
        pathway_data_grouped.append({
            'category': category,
            'start_angle': category_start_angle,
            'end_angle': current_angle,
            'data': category_data
        })

    # Create a new dataframe with all the angle-assigned data
    pathway_data = pd.concat([group_info['data'] for group_info in pathway_data_grouped])

    # Set up the polar plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, ylim)

    # Add the axis from the center
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_linewidth(2)
    ax.spines['polar'].set_color('black')

    # Set the position of the score ticks
    ax.set_rlabel_position(180)
    ax.set_yticks([np.min([1., ylim]), ylim])
    ax.set_yticklabels(['', ''], fontsize=16)
    ax.tick_params(axis='y', which='major', width=1., color='red')

    # Color the circular area behind each category with exact alignment
    for group_info in pathway_data_grouped:
        category = group_info['category']
        start_angle = group_info['start_angle']
        end_angle = group_info['end_angle']

        # Create angles that span exactly from the first to last pathway in this category
        cat_angles = np.linspace(start_angle, end_angle, 50)

        # Fill the background
        ax.fill_between(cat_angles, 0, ylim, color=color_map[category], alpha=alpha_bg, zorder=0)

    # Plot the data on the radial plot with connected bars within each category
    for group_info in pathway_data_grouped:
        category = group_info['category']

        # Get all pathways in this category
        category_paths = pathway_data[pathway_data[category_col] == category]

        # Get angles and scores for this category
        cat_angles = category_paths['angle'].values
        cat_scores = category_paths[value_col].values

        # For each pathway in the category, draw the radial line
        for angle, score in zip(cat_angles, cat_scores):
            ax.plot([angle, angle], [0, score], color=color_map[category], linewidth=2)

        # Connect all bars within this category with a polygon
        polygon_angles = []
        polygon_radii = []

        # Start the polygon at the first bar, at the base
        for angle, score in zip(cat_angles, cat_scores):
            # Add a point at the base (radius 0)
            polygon_angles.append(angle)
            polygon_radii.append(0)

            # Add a point at the top of the bar
            polygon_angles.append(angle)
            polygon_radii.append(score)

        # Convert to numpy arrays for matplotlib
        polygon_angles = np.array(polygon_angles)
        polygon_radii = np.array(polygon_radii)

        # Plot the polygon
        ax.fill(polygon_angles, polygon_radii, color=color_map[category], alpha=alpha_fill)

    # Remove theta ticks
    ax.set_xticks([])

    # Create a legend only if show_legend is True
    if show_legend:
        # Create a legend using the same category order as the colors
        legend_labels = categories_by_size
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[cat], lw=0) for cat in legend_labels]
        legend_labels = [l.upper() for l in legend_labels]

        # Create the legend
        ax.legend(legend_handles, legend_labels,
                  loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor,
                  fontsize=legend_fontsize, frameon=False, borderaxespad=0,
                  ncol=1)

    # Add title with appropriate information (if not None)
    if title is not None:
        title_parts = []
        title_parts.append(title)

        if tissue is not None:
            title_parts.append(tissue)

        if cell_type is not None:
            title_parts.append(cell_type)
        else:
            title_parts.append("across cell types")

        # Join title parts with newlines
        full_title = '\n'.join(title_parts)

        # Use set_title if this is a subplot, otherwise use suptitle
        if ax is not None and ax.get_subplotspec() is not None:
            ax.set_title(full_title, fontsize=title_fontsize, fontweight='bold', pad=20)
        else:
            plt.suptitle(full_title, fontsize=title_fontsize, fontweight='bold', y=1.025)

    # Apply tight layout if requested and we created the figure
    if tight_layout and ax is None:
        plt.tight_layout()

    # Save if requested
    if save is not None:
        try:
            from sccellfie.plotting.plot_utils import _get_file_format, _get_file_dir
            dir, basename = _get_file_dir(save)
            os.makedirs(dir, exist_ok=True)
            format = _get_file_format(save)
            plt.savefig(f'{dir}/radial_{basename}.{format}', dpi=dpi, bbox_inches=bbox_inches)
        except ImportError:
            # Fall back to basic save if plot_utils is not available
            plt.savefig(save, dpi=dpi, bbox_inches=bbox_inches)

    return fig, ax