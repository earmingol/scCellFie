import pytest
import numpy as np
import pandas as pd

from sccellfie.external.tf_idf import quick_markers, filter_tfidf_markers
from sccellfie.datasets.toy_inputs import create_random_adata


def test_output_structure():
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster')

    expected_columns = ['gene', 'cluster', 'tf', 'idf', 'tf_idf', 'gene_frequency_outside_cluster',
                        'gene_frequency_global', 'second_best_tf', 'second_best_cluster', 'pval', 'qval']
    assert isinstance(markers, pd.DataFrame), "Output is not a pandas DataFrame"
    assert all(column in markers.columns for column in expected_columns), "Missing columns in output"


def test_number_of_markers():
    n_markers = 5
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster', n_markers=n_markers, fdr=1.0)

    for cluster in markers['cluster'].unique():
        assert len(markers[markers['cluster'] == cluster]) == n_markers


def test_fdr_filtering():
    fdr = 0.999999
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster', fdr=fdr)

    if markers.shape[0] > 0:
        assert all(markers['qval'] < fdr)


def test_expression_cutoff():
    express_cut = 10
    adata = create_random_adata()
    markers1 = quick_markers(adata, cluster_key='cluster', fdr=1.0)
    markers2 = quick_markers(adata, cluster_key='cluster', express_cut=express_cut, fdr=1.0)

    assert ~markers1.equals(markers2), "The two marker sets are the same, expression cutoff not making a difference"


def test_quick_markers_r_output_columns():
    r_output = True
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster', r_output=r_output)

    expected_columns = ['gene', 'cluster', 'geneFrequency', 'geneFrequencyOutsideCluster',
                        'geneFrequencySecondBest', 'geneFrequencyGlobal', 'secondBestClusterName', 'tfidf', 'idf', 'qval']
    assert all(column in markers.columns for column in expected_columns), "Missing columns in output"


def test_cell_groups():
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster', cell_groups=['cluster1', 'cluster2', 'cluster3'], fdr=1.0)

    expected_columns = ['gene', 'cluster', 'tf', 'idf', 'tf_idf', 'gene_frequency_outside_cluster',
                        'gene_frequency_global', 'second_best_tf', 'second_best_cluster', 'pval', 'qval']
    assert isinstance(markers, pd.DataFrame), "Output is not a pandas DataFrame"
    assert all(column in markers.columns for column in expected_columns), "Missing columns in output"
    assert len(markers['cluster'].unique().tolist()) == len(['cluster1', 'cluster2', 'cluster3']), "Clusters are not properly filtered"
    assert all(c in markers['cluster'].unique() for c in ['cluster1', 'cluster2', 'cluster3']), "Missing clusters in output"
    assert all(c not in markers['cluster'].unique() for c in ['cluster4', 'cluster5']), "Missing clusters in output"


@pytest.fixture
def sample_marker_df():
    """Create a sample DataFrame with marker data including cluster information"""
    data = {
        'tf': [0.8, 0.6, 0.4, 0.2, 0.1],
        'idf': [0.9, 0.7, 0.5, 0.3, 0.2],
        'tf_idf': [0.72, 0.42, 0.2, 0.06, 0.02],
        'second_best_tf': [0.6, 0.4, 0.3, 0.15, 0.08],
        'cluster': ['A', 'A', 'B', 'B', 'C'],
        'second_best_cluster': ['B', 'C', 'A', 'C', 'A']
    }
    return pd.DataFrame(data)


def test_filter_tfidf_basic(sample_marker_df):
    """Test basic functionality without additional filters"""
    filtered_df, theoretical_curve = filter_tfidf_markers(sample_marker_df)

    # Check if output types are correct
    assert isinstance(filtered_df, pd.DataFrame)
    assert isinstance(theoretical_curve, tuple)
    assert len(theoretical_curve) == 2

    # Check if theoretical curve has expected properties
    x_curve, y_curve = theoretical_curve
    assert len(x_curve) == len(sample_marker_df)
    assert len(y_curve) == len(sample_marker_df)
    assert np.all(np.isfinite(y_curve))


def test_filter_tfidf_threshold(sample_marker_df):
    """Test filtering with TF-IDF threshold"""
    tfidf_threshold = 0.3
    filtered_df, _ = filter_tfidf_markers(sample_marker_df, tfidf_threshold=tfidf_threshold)

    # Check if all remaining entries meet the threshold
    assert np.all(filtered_df['tf_idf'] > tfidf_threshold)


def test_filter_tfidf_ratio(sample_marker_df):
    """Test filtering with TF ratio threshold"""
    tf_ratio = 1.2
    filtered_df, _ = filter_tfidf_markers(sample_marker_df, tf_ratio=tf_ratio)

    # Check if all remaining entries meet either the ratio threshold or have matching clusters
    assert np.all(
        (filtered_df['tf'] / filtered_df['second_best_tf'] > tf_ratio) |
        (filtered_df['cluster'] == filtered_df['second_best_cluster'])
    )


def test_filter_tfidf_cluster_matching(sample_marker_df):
    """Test filtering when clusters match"""
    # Modify the sample data to include matching clusters
    df_with_matching = sample_marker_df.copy()
    df_with_matching.loc[0, 'second_best_cluster'] = df_with_matching.loc[0, 'cluster']

    tf_ratio = 2.0  # High ratio that would normally filter out the first row
    filtered_df, _ = filter_tfidf_markers(df_with_matching, tf_ratio=tf_ratio)

    # Check if the row with matching clusters is retained despite not meeting the ratio threshold
    matching_clusters_mask = df_with_matching['cluster'] == df_with_matching['second_best_cluster']
    assert not np.all(filtered_df['tf'] / filtered_df['second_best_tf'] > tf_ratio)
    assert any(matching_clusters_mask & filtered_df.index.isin(df_with_matching.index))


def test_filter_tfidf_combined(sample_marker_df):
    """Test filtering with both TF-IDF and ratio thresholds"""
    tfidf_threshold = 0.3
    tf_ratio = 1.2
    filtered_df, _ = filter_tfidf_markers(
        sample_marker_df,
        tfidf_threshold=tfidf_threshold,
        tf_ratio=tf_ratio
    )

    # Check if all remaining entries meet the TF-IDF threshold
    assert np.all(filtered_df['tf_idf'] > tfidf_threshold)

    # Check if all remaining entries meet either the ratio threshold or have matching clusters
    assert np.all(
        (filtered_df['tf'] / filtered_df['second_best_tf'] > tf_ratio) |
        (filtered_df['cluster'] == filtered_df['second_best_cluster'])
    )


def test_filter_tfidf_custom_columns(sample_marker_df):
    """Test filtering with custom column names"""
    # Rename columns
    df_custom = sample_marker_df.rename(columns={
        'tf': 'custom_tf',
        'idf': 'custom_idf',
        'tf_idf': 'custom_tfidf',
        'second_best_tf': 'custom_second_best',
        'cluster': 'custom_cluster',
        'second_best_cluster': 'custom_second_best_cluster'
    })

    filtered_df, _ = filter_tfidf_markers(
        df_custom,
        tf_col='custom_tf',
        idf_col='custom_idf',
        tfidf_col='custom_tfidf',
        second_best_tf_col='custom_second_best',
        group_col='custom_cluster',
        second_best_group_col='custom_second_best_cluster'
    )

    # Check if function works with custom column names
    assert isinstance(filtered_df, pd.DataFrame)
    assert len(filtered_df) > 0


def test_filter_tfidf_theoretical_curve_properties(sample_marker_df):
    """Test properties of the theoretical curve"""
    _, (x_curve, y_curve) = filter_tfidf_markers(sample_marker_df)

    # Check if curve x values span the input range
    assert np.min(x_curve) <= np.min(sample_marker_df['tf'])
    assert np.max(x_curve) >= np.max(sample_marker_df['tf'])

    # Check if curve values are monotonic (should be decreasing for hyperbola)
    assert np.all(np.diff(y_curve) <= 0)


@pytest.mark.parametrize("invalid_threshold", [-1.0, 2.0])
def test_filter_tfidf_invalid_threshold(sample_marker_df, invalid_threshold):
    """Test handling of invalid threshold values"""
    filtered_df, _ = filter_tfidf_markers(sample_marker_df, tfidf_threshold=invalid_threshold)

    # Function should still run and return valid output
    assert isinstance(filtered_df, pd.DataFrame)
    assert len(filtered_df) >= 0  # May be empty but should exist


def test_filter_tfidf_empty_result(sample_marker_df):
    """Test filtering with strict thresholds that result in empty DataFrame"""
    filtered_df, theoretical_curve = filter_tfidf_markers(
        sample_marker_df,
        tfidf_threshold=1.0,  # Higher than all values
        tf_ratio=10.0  # Higher than all ratios
    )

    # Check if result is empty but maintains structure
    assert len(filtered_df) == 0
    assert list(filtered_df.columns) == list(sample_marker_df.columns)