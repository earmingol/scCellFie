import pandas as pd

from sccellfie.external.tf_idf import quick_markers
from sccellfie.tests.toy_inputs import create_random_adata


def test_output_structure():
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster')

    expected_columns = ['gene', 'cluster', 'tf', 'idf', 'tf_idf', 'gene_frequency_outside_cluster',
                        'gene_frequency_global', 'second_best_tf', 'second_best_cluster', 'pval', 'qval']
    assert isinstance(markers, pd.DataFrame)
    assert all(column in markers.columns for column in expected_columns)


def test_number_of_markers():
    n_markers = 5
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster', n_markers=n_markers, fdr=1.0)

    for cluster in markers['cluster'].unique():
        assert len(markers[markers['cluster'] == cluster]) == n_markers


def test_fdr_filtering():
    fdr = 0.05
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster', fdr=fdr)

    if markers.shape[0] > 0:
        assert all(markers['qval'] < fdr)


def test_expression_cutoff():
    express_cut = 10
    adata = create_random_adata()
    markers1 = quick_markers(adata, cluster_key='cluster', fdr=1.0)
    markers2 = quick_markers(adata, cluster_key='cluster', express_cut=express_cut, fdr=1.0)

    assert ~markers1.equals(markers2)


def test_quick_markers_r_output_columns():
    r_output = True
    adata = create_random_adata()
    markers = quick_markers(adata, cluster_key='cluster', r_output=r_output)

    expected_columns = ['gene', 'cluster', 'geneFrequency', 'geneFrequencyOutsideCluster',
                        'geneFrequencySecondBest', 'geneFrequencyGlobal', 'secondBestClusterName', 'tfidf', 'idf', 'qval']
    assert all(column in markers.columns for column in expected_columns)