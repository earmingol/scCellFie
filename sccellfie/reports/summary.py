import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from sccellfie.expression.aggregation import AGG_FUNC


def generate_report_from_adata(adata, group_by, agg_func='trimean', layer=None, features=None, tissue_col=None,
                               feature_name='feature', min_cells=1, threshold=5 * np.log(2),
                               default_tissue_name='tissue', **kwargs):
    """
    Process AnnData object and calculate metrics for each group (e.g., cell type).

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the expression data.

    group_by : str
        Column name in adata.obs for the groups (e.g., cell types).

    agg_func : str, optional (default: 'trimean')
        The aggregation function to apply. Options are 'mean', 'median',
        '25p' (25th percentile), '75p' (75th percentile), 'trimean' (0.5*Q2 + 0.25(Q1+Q3)),
        and 'topmean' (computed among the top `top_percent`% of values).

    layer : str, optional (default: None)
        Name of the layer in adata to use. If None, uses adata.X.

    features : list, optional (default: None)
        Names of features to analyze. If None, uses adata.var_names.

    tissue_col : str, optional (default: None)
        Column name in adata.obs for tissue information.

    feature_name : str, optional (default: 'feature')
        Name to use for features in melted results (e.g., 'metabolic_task', 'reaction').

    min_cells : int, optional (default: 1)
        Minimum number of cells required for a group to be included in the analysis.

    threshold : float, optional (default: 5*np.log(2))
        Threshold value for counting cells passing expression threshold.

    default_tissue_name : str, optional (default: 'tissue')
        Default tissue name to use when tissue_column is not provided.

    **kwargs : dict
        Additional arguments to pass to the aggregation function.

    Returns
    -------
    dict
        Dictionary containing DataFrames for each metric:
            - agg_values: Aggregated values (e.g., trimean) per group
            - variance: Variance values per group
            - std: Standard deviation values per group
            - threshold_cells: Number of cells passing threshold per group
            - nonzero_cells: Number of non-zero cells per group
            - cell_counts: Number of cells per group
            - min_max: Min/max values for features
            - melted: Melted version of all metrics
    """
    # Get the data matrix
    if layer is None:
        data_matrix = adata.X
    else:
        data_matrix = adata.layers[layer]

    # Get feature names
    if features is None:
        if hasattr(adata, 'var_names'):
            feature_names = adata.var_names
        else:
            # Use index numbers as feature names if not available
            feature_names = [f'Feature_{i}' for i in range(data_matrix.shape[1])]
    else:
        feature_names = features

    # Extract metrics by group
    metrics_data = generate_summary_by_group(
        adata, data_matrix, feature_names, group_by, agg_func,
        tissue_col, min_cells, default_tissue_name, threshold, **kwargs
    )

    # Compile feature ranges
    feature_ranges = summarize_feature_range(metrics_data['features_seen'],
                                             metrics_data['single_cell_min'],
                                             metrics_data['single_cell_max'],
                                             metrics_data['group_min'],
                                             metrics_data['group_max'])

    # Aggregate DataFrames
    result_dfs = aggregate_metrics_dataframes(metrics_data['results'])

    # Create melted results
    melted_results = melt_summary_data(
        result_dfs['agg_values'], result_dfs['variance'],
        result_dfs['std'], result_dfs['threshold_cells'],
        result_dfs['nonzero_cells'], result_dfs['cell_counts_df'],
        feature_name, agg_func
    )

    ## Add scaled results
    # Apply the mapping in a vectorized way
    melted_results['min_value'] = melted_results[feature_name].map(metrics_data['group_min'])
    melted_results['max_value'] = melted_results[feature_name].map(metrics_data['group_max'])

    # Perform the calculation
    melted_results[f'scaled_{agg_func}'] = (melted_results[agg_func] - melted_results['min_value']) / (
            melted_results['max_value'] - melted_results['min_value'])

    # Clean up intermediate columns if desired
    melted_results = melted_results.drop(['min_value', 'max_value'], axis=1)
    melted_results = melted_results[
        [feature_name, 'tissue', 'cell_type', agg_func, f'scaled_{agg_func}', 'variance', 'std', 'n_cells_threshold',
         'n_cells_nonzero', 'total_cells']]

    # Compile final results
    return {
        'agg_values': result_dfs['agg_values'],
        'variance': result_dfs['variance'],
        'std': result_dfs['std'],
        'threshold_cells': result_dfs['threshold_cells'],
        'nonzero_cells': result_dfs['nonzero_cells'],
        'cell_counts': result_dfs['cell_counts_df'],
        'min_max': feature_ranges,
        'melted': melted_results
    }


def generate_summary_by_group(adata, data_matrix, feature_names, group_by, agg_func,
                              tissue_column, min_cells, default_tissue_name, threshold, **kwargs):
    """
    Generate metrics for each group/tissue combination.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the expression data.

    data_matrix : numpy.ndarray or scipy.sparse matrix
        Expression data matrix.

    feature_names : list
        Names of features to analyze.

    group_by : str
        Column name in adata.obs for the groups.

    agg_func : str
        The aggregation function to apply.

    tissue_column : str or None
        Column name in adata.obs for tissue information.

    min_cells : int
        Minimum number of cells required for a group to be included.

    default_tissue_name : str
        Default tissue name to use when tissue_column is not provided.

    threshold : float
        Threshold value for counting cells passing expression threshold.

    **kwargs : dict
        Additional arguments to pass to the aggregation function.

    Returns
    -------
    dict
        Dictionary containing intermediate results.
    """
    # Get unique groups
    unique_groups = adata.obs[group_by].unique()

    # Get unique tissues if tissue_column is provided, otherwise use default
    if tissue_column is not None and tissue_column in adata.obs.columns:
        unique_tissues = adata.obs[tissue_column].unique()
        has_tissue_col = True
    else:
        unique_tissues = [default_tissue_name]
        has_tissue_col = False

    # Initialize data structures
    results = []
    single_cell_min = {}
    single_cell_max = {}
    group_min = {}
    group_max = {}
    features_seen = set()

    # Process each tissue and group combination
    for tissue in tqdm(unique_tissues, desc='Processing tissues'):
        if has_tissue_col:
            # Filter by tissue
            tissue_mask = adata.obs[tissue_column] == tissue
            tissue_name = tissue
        else:
            # No tissue filtering, use all cells with default tissue name
            tissue_mask = np.ones(adata.n_obs, dtype=bool)
            tissue_name = tissue  # This is default_tissue_name

        for group in tqdm(unique_groups, desc=f'Processing groups for {tissue_name}', leave=False):
            # Create mask for this group (and tissue if applicable)
            group_mask = adata.obs[group_by] == group
            combined_mask = group_mask & tissue_mask

            # Skip if no cells match
            if np.sum(combined_mask) == 0:
                continue

            # Get subset of data
            subset_data = data_matrix[combined_mask]

            # If the data is sparse, convert to dense for calculations
            if sparse.issparse(subset_data):
                subset_data = subset_data.toarray()

            # Create a DataFrame for easier processing
            df = pd.DataFrame(subset_data, columns=feature_names)

            # Track all features seen
            features_seen.update(df.columns)

            # Calculate metrics
            n_cells = df.shape[0]
            col_name = f'{tissue_name} / {group}'

            if n_cells < min_cells:
                continue

            # Calculate metrics for this group
            group_metrics = calculate_group_summary(df, col_name, agg_func, threshold, **kwargs)

            # Store results with tissue information
            results.append((tissue_name, group, group_metrics['agg_df'], group_metrics['variance_df'],
                            group_metrics['std_df'], group_metrics['threshold_df'],
                            group_metrics['nonzero_df'], n_cells))

            # Update global min/max tracking
            update_min_max_values(
                df, group_metrics['agg_values'], features_seen,
                single_cell_min, single_cell_max,
                group_min, group_max
            )

    return {
        'results': results,
        'single_cell_min': single_cell_min,
        'single_cell_max': single_cell_max,
        'group_min': group_min,
        'group_max': group_max,
        'features_seen': features_seen
    }


def calculate_group_summary(df, col_name, agg_func, threshold=5 * np.log(2), **kwargs):
    """
    Calculate metrics for a specific group.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing expression data for a specific group.

    col_name : str
        Name to use for the column in the result DataFrames.

    agg_func : str
        The aggregation function to apply.

    threshold : float, optional (default: 5*np.log(2))
        Threshold value for counting cells passing expression threshold.

    **kwargs : dict
        Additional arguments to pass to the aggregation function.

    Returns
    -------
    dict
        Dictionary containing metrics for this group.
    """
    # Calculate aggregated values (e.g., trimean)
    agg_values = AGG_FUNC[agg_func](df, axis=0, **kwargs)
    agg_df = pd.DataFrame(
        agg_values,
        index=df.columns,
        columns=[col_name]
    )

    # Calculate variance
    variance_values = df.var(axis=0)
    variance_df = pd.DataFrame(
        variance_values,
        index=df.columns,
        columns=[col_name]
    )

    # Calculate standard deviation
    std_values = df.std(axis=0)
    std_df = pd.DataFrame(
        std_values,
        index=df.columns,
        columns=[col_name]
    )

    # Count cells passing expression threshold (greater than or equal to the threshold parameter)
    threshold_df = (df.ge(threshold).astype(int).sum()).to_frame()
    threshold_df.columns = [col_name]

    # Count cells with non-zero values
    nonzero_df = (df.gt(0).astype(int).sum()).to_frame()
    nonzero_df.columns = [col_name]

    return {
        'agg_values': agg_values,
        'agg_df': agg_df,
        'variance_df': variance_df,
        'std_df': std_df,
        'threshold_df': threshold_df,
        'nonzero_df': nonzero_df
    }


def update_min_max_values(df, agg_values, features_seen,
                          single_cell_min, single_cell_max,
                          group_min, group_max):
    """
    Update min/max tracking dictionaries with values from a new group.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing expression data for a specific group.

    agg_values : numpy.ndarray
        Aggregated values for the group.

    features_seen : set
        Set of all features seen so far.

    single_cell_min : dict
        Dictionary tracking minimum single-cell value for each feature.

    single_cell_max : dict
        Dictionary tracking maximum single-cell value for each feature.

    group_min : dict
        Dictionary tracking minimum group-aggregated value for each feature.

    group_max : dict
        Dictionary tracking maximum group-aggregated value for each feature.
    """
    # Get min/max for all features using pandas operations
    curr_min_series = df.min()
    curr_max_series = df.max()
    curr_agg_series = pd.Series(agg_values, index=df.columns)

    # Initialize dictionaries if they're empty
    if not single_cell_min:
        single_cell_min.update(curr_min_series.to_dict())
    if not single_cell_max:
        single_cell_max.update(curr_max_series.to_dict())
    if not group_min:
        group_min.update(curr_agg_series.to_dict())
    if not group_max:
        group_max.update(curr_agg_series.to_dict())
    else:
        # Update global min/max values for features in this group
        for feature in df.columns:
            # Update min/max for single cells
            single_cell_min[feature] = min(
                single_cell_min.get(feature, float('inf')),
                curr_min_series[feature]
            )
            single_cell_max[feature] = max(
                single_cell_max.get(feature, float('-inf')),
                curr_max_series[feature]
            )

            # Update min/max for group aggregates
            group_min[feature] = min(
                group_min.get(feature, float('inf')),
                curr_agg_series[feature]
            )
            group_max[feature] = max(
                group_max.get(feature, float('-inf')),
                curr_agg_series[feature]
            )


def summarize_feature_range(features_seen, single_cell_min, single_cell_max,
                            group_min, group_max):
    """
    Create a DataFrame summarizing min/max values for features.

    Parameters
    ----------
    features_seen : set
        Set of all features seen.

    single_cell_min : dict
        Dictionary tracking minimum single-cell value for each feature.

    single_cell_max : dict
        Dictionary tracking maximum single-cell value for each feature.

    group_min : dict
        Dictionary tracking minimum group-aggregated value for each feature.

    group_max : dict
        Dictionary tracking maximum group-aggregated value for each feature.

    Returns
    -------
    pandas.DataFrame
        DataFrame summarizing min/max values for features.
    """
    # Create min/max DataFrame
    min_max_rows = ['single_cell_min', 'single_cell_max', 'group_min', 'group_max']
    min_max_data = {}

    for feature in sorted(features_seen):
        min_max_data[feature] = [
            single_cell_min.get(feature, 0),
            single_cell_max.get(feature, 0),
            group_min.get(feature, 0),
            group_max.get(feature, 0)
        ]

    return pd.DataFrame(min_max_data, index=min_max_rows)


def aggregate_metrics_dataframes(results):
    """
    Aggregate metrics DataFrames from individual groups.

    Parameters
    ----------
    results : list
        List of tuples containing metrics for each group.

    Returns
    -------
    dict
        Dictionary containing aggregated DataFrames.
    """
    # Extract components from results
    tissues = []
    cell_types = []
    agg_dfs = []
    variance_dfs = []
    std_dfs = []
    threshold_cells_dfs = []
    nonzero_cells_dfs = []
    cell_counts = []

    for tissue, cell_type, agg_df, variance_df, std_df, threshold_cells_df, nonzero_cells_df, n_cells in results:
        tissues.append(tissue)
        cell_types.append(cell_type)
        agg_dfs.append(agg_df)
        variance_dfs.append(variance_df)
        std_dfs.append(std_df)
        threshold_cells_dfs.append(threshold_cells_df)
        nonzero_cells_dfs.append(nonzero_cells_df)
        cell_counts.append((tissue, cell_type, n_cells))

    # Concatenate the DataFrames
    agg_values = pd.concat(agg_dfs, axis=1).fillna(0) if agg_dfs else pd.DataFrame()
    variance = pd.concat(variance_dfs, axis=1).fillna(0) if variance_dfs else pd.DataFrame()
    std = pd.concat(std_dfs, axis=1).fillna(0) if std_dfs else pd.DataFrame()
    threshold_cells = pd.concat(threshold_cells_dfs, axis=1).fillna(0) if threshold_cells_dfs else pd.DataFrame()
    nonzero_cells = pd.concat(nonzero_cells_dfs, axis=1).fillna(0) if nonzero_cells_dfs else pd.DataFrame()

    # Create cell counts DataFrame
    cell_counts_df = create_cell_counts_df(cell_counts)

    return {
        'agg_values': agg_values,
        'variance': variance,
        'std': std,
        'threshold_cells': threshold_cells,
        'nonzero_cells': nonzero_cells,
        'cell_counts_df': cell_counts_df
    }


def create_cell_counts_df(cell_counts):
    """
    Create DataFrame with cell counts for each group.

    Parameters
    ----------
    cell_counts : list
        List of tuples (tissue, cell_type, n_cells)

    Returns
    -------
    pandas.DataFrame
        DataFrame with cell counts.
    """
    # Create DataFrame with tissue, cell_type, and total_cells columns
    return pd.DataFrame(
        cell_counts,
        columns=['tissue', 'cell_type', 'total_cells']
    ) if cell_counts else pd.DataFrame(columns=['tissue', 'cell_type', 'total_cells'])


def melt_summary_data(agg_values, variance, std, threshold_cells, nonzero_cells,
                      cell_counts_df, feature_name='feature', agg_func='trimean'):
    """
    Melt metrics DataFrames into a long format for easier visualization and analysis.

    Parameters
    ----------
    agg_values : pandas.DataFrame
        DataFrame containing aggregated values (e.g., trimean).

    variance : pandas.DataFrame
        DataFrame containing variance values.

    std : pandas.DataFrame
        DataFrame containing standard deviation values.

    threshold_cells : pandas.DataFrame
        DataFrame containing threshold cells counts.

    nonzero_cells : pandas.DataFrame
        DataFrame containing nonzero cells counts.

    cell_counts_df : pandas.DataFrame
        DataFrame containing cell counts by group.

    feature_name : str, optional (default: 'feature')
        Name to use for the feature column (e.g., 'metabolic_task', 'reaction').

    agg_func : str, optional (default: 'trimean')
        Name of the aggregation function used.

    Returns
    -------
    pandas.DataFrame
        Long format dataframe containing all metrics.
    """
    if agg_values.empty:
        return pd.DataFrame()

    # Create a dictionary to map tissue+cell_type to total cells
    cell_counts_df['tissue_celltype'] = cell_counts_df['tissue'] + ' / ' + cell_counts_df['cell_type']
    total_cells_dict = dict(zip(cell_counts_df['tissue_celltype'], cell_counts_df['total_cells']))

    # Create a mapping from combined key to individual tissue and cell_type
    tissue_dict = dict(zip(cell_counts_df['tissue_celltype'], cell_counts_df['tissue']))
    cell_type_dict = dict(zip(cell_counts_df['tissue_celltype'], cell_counts_df['cell_type']))

    # Initialize lists to store the melted data
    data = []

    # Get features
    features = agg_values.index

    # Iterate through the columns (tissue/cell type combinations)
    for col in agg_values.columns:
        tissue = tissue_dict.get(col)
        cell_type = cell_type_dict.get(col)
        total_cells = total_cells_dict.get(col, 0)

        # Iterate through features
        for feature in features:
            entry = {
                feature_name: feature,
                'tissue': tissue,
                'cell_type': cell_type,
                agg_func: agg_values.loc[feature, col],
                'variance': variance.loc[feature, col],
                'std': std.loc[feature, col],
                'n_cells_threshold': threshold_cells.loc[feature, col],
                'n_cells_nonzero': nonzero_cells.loc[feature, col],
                'total_cells': total_cells
            }

            data.append(entry)

    # Create dataframe
    df_melted = pd.DataFrame(data)

    # Sort the dataframe
    df_melted = df_melted.sort_values(['tissue', 'cell_type', feature_name]).reset_index(drop=True)

    return df_melted