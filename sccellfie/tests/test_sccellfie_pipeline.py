import pytest
import numpy as np
import pandas as pd
from anndata import AnnData

from sccellfie.sccellfie_pipeline import run_sccellfie_pipeline, compute_neighbors_pipeline
from sccellfie.datasets.toy_inputs import (create_random_adata, create_controlled_adata, create_controlled_gpr_dict,
                                           create_controlled_task_by_rxn, create_controlled_rxn_by_gene, create_controlled_task_by_gene,
                                           create_global_threshold, add_toy_neighbors)


@pytest.fixture
def random_adata_with_neighbors():
    adata = create_random_adata(n_obs=100, n_vars=3, n_clusters=5)
    adata = add_toy_neighbors(adata)
    return adata


@pytest.fixture
def controlled_adata_with_neighbors():
    adata = create_controlled_adata()
    adata = add_toy_neighbors(adata)
    return adata


@pytest.fixture
def controlled_gpr_dict():
    gpr_dict = create_controlled_gpr_dict()
    # Convert to DataFrame
    return pd.DataFrame([{'Reaction': k, 'GPR-symbol': v.to_string()} for k, v in gpr_dict.items()])


@pytest.fixture
def controlled_task_by_rxn():
    return create_controlled_task_by_rxn()


@pytest.fixture
def controlled_rxn_by_gene():
    return create_controlled_rxn_by_gene()


@pytest.fixture
def controlled_task_by_gene():
    return create_controlled_task_by_gene()


@pytest.fixture
def global_threshold():
    df = create_global_threshold(threshold=0.5, n_vars=3)
    df.index.name = 'symbol'
    return df


@pytest.fixture
def sccellfie_db(controlled_gpr_dict, controlled_task_by_rxn, controlled_rxn_by_gene, controlled_task_by_gene,
                 global_threshold):
    db = {
        'rxn_info': controlled_gpr_dict,
        'task_by_rxn': controlled_task_by_rxn,
        'rxn_by_gene': controlled_rxn_by_gene,
        'task_by_gene': controlled_task_by_gene,
        'thresholds': global_threshold
    }
    return db


def test_run_sccellfie_pipeline(random_adata_with_neighbors, sccellfie_db, tmp_path, monkeypatch):
    result = run_sccellfie_pipeline(
        adata=random_adata_with_neighbors,
        organism='human',
        sccellfie_data_folder=None,
        sccellfie_db=sccellfie_db,
        n_counts_col='total_counts',
        process_by_group=False,
        groupby=None,
        neighbors_key='neighbors',
        n_neighbors=10,
        batch_key=None,
        threshold_key='global',
        smooth_cells=True,
        alpha=0.33,
        chunk_size=5000,
        disable_pbar=True,
        save_folder=str(tmp_path),
        save_filename='test_output',
        verbose=True  # Set to True to see more output
    )

    assert isinstance(result, dict)
    assert 'adata' in result
    assert isinstance(result['adata'], AnnData)
    assert 'gene_scores' in result['adata'].layers
    assert hasattr(result['adata'], 'reactions')
    assert hasattr(result['adata'], 'metabolic_tasks')


def test_run_sccellfie_pipeline_with_groups(random_adata_with_neighbors, sccellfie_db, tmp_path, monkeypatch):
    random_adata_with_neighbors.obs['group'] = np.random.choice(['A', 'B', 'C'], size=random_adata_with_neighbors.n_obs)
    
    run_sccellfie_pipeline(
        adata=random_adata_with_neighbors,
        organism='human',
        sccellfie_data_folder=None,
        sccellfie_db=sccellfie_db,
        n_counts_col='total_counts',  # Changed from 'n_counts' to match the fixture
        process_by_group=True,
        groupby='group',
        neighbors_key='neighbors',
        n_neighbors=10,
        batch_key=None,
        threshold_key='global',
        smooth_cells=True,
        alpha=0.33,
        chunk_size=5000,
        disable_pbar=True,
        save_folder=str(tmp_path),
        save_filename='test_output_group',
        verbose=False
    )
    
    # Check if files were created for each group
    for group in ['A', 'B', 'C']:
        assert (tmp_path / f"test_output_group_{group}.h5ad").exists()


# Neighbors pipeline
@pytest.fixture
def mock_adata():
    # Create smaller dataset with more realistic values
    n_obs = 50
    n_vars = 30
    X = np.random.negative_binomial(20, 0.3, size=(n_obs, n_vars))
    adata = AnnData(X)
    adata.obs['batch'] = ['A'] * (n_obs // 2) + ['B'] * (n_obs // 2)
    adata.var_names = [f'gene{i}' for i in range(n_vars)]
    return adata


def test_compute_neighbors_pipeline_basic(mock_adata):
    compute_neighbors_pipeline(mock_adata, batch_key=None, n_neighbors=5)
    assert 'neighbors' in mock_adata.uns
    assert 'X_pca' in mock_adata.obsm
    assert 'X_umap' in mock_adata.obsm


# @pytest.mark.skipif(
#    pytest.importorskip("harmonypy", reason="harmony not installed"),
#    reason="harmony not installed"
# )
# def test_compute_neighbors_pipeline_with_harmony(mock_adata):
#    compute_neighbors_pipeline(mock_adata, batch_key='batch', n_neighbors=5)
#    assert 'neighbors' in mock_adata.uns
#    assert 'X_pca' in mock_adata.obsm
#    assert 'X_pca_harmony' in mock_adata.obsm


def test_compute_neighbors_pipeline_without_harmony(mock_adata):
   compute_neighbors_pipeline(mock_adata, batch_key='batch', n_neighbors=5)
   assert 'neighbors' in mock_adata.uns
   assert 'X_pca' in mock_adata.obsm


def test_compute_neighbors_pipeline_custom_n_neighbors(mock_adata):
    n_neighbors = 15
    compute_neighbors_pipeline(mock_adata, batch_key=None, n_neighbors=n_neighbors)
    assert mock_adata.uns['neighbors']['params']['n_neighbors'] == n_neighbors


def test_compute_neighbors_pipeline_existing_normalization(mock_adata):
    mock_adata.uns['normalization'] = True
    compute_neighbors_pipeline(mock_adata, batch_key=None)
    assert 'neighbors' in mock_adata.uns
    assert 'X_pca' in mock_adata.obsm


def test_compute_neighbors_pipeline_existing_umap(mock_adata):
    mock_adata.obsm['X_umap'] = np.random.normal(size=(mock_adata.n_obs, 2))
    original_umap = mock_adata.obsm['X_umap'].copy()
    compute_neighbors_pipeline(mock_adata, batch_key=None)
    assert np.allclose(mock_adata.obsm['X_umap'], original_umap)