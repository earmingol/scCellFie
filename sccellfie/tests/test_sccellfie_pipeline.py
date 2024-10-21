import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from sccellfie.sccellfie_pipeline import run_sccellfie_pipeline, process_chunk, compute_neighbors
from sccellfie.datasets.toy_inputs import (create_random_adata, create_controlled_adata, create_controlled_gpr_dict,
                                           create_controlled_task_by_rxn, create_controlled_rxn_by_gene, create_controlled_task_by_gene,
                                           create_global_threshold)


def add_toy_neighbors(adata, n_neighbors=10):
    """
    Add a toy neighbor object to the AnnData object, mimicking scanpy's format.
    """
    n_obs = adata.n_obs
    
    # Create toy connectivities and distances matrices
    connectivities = csr_matrix((np.ones(n_obs * n_neighbors),
                                 (np.repeat(np.arange(n_obs), n_neighbors),
                                  np.random.choice(n_obs, n_obs * n_neighbors))),
                                shape=(n_obs, n_obs))
    
    distances = csr_matrix((np.random.rand(n_obs * n_neighbors),
                            (np.repeat(np.arange(n_obs), n_neighbors),
                             np.random.choice(n_obs, n_obs * n_neighbors))),
                           shape=(n_obs, n_obs))
    
    # Create the neighbors dictionary
    adata.uns['neighbors'] = {
        'params': {
            'n_neighbors': n_neighbors,
            'method': 'umap'
        },
        'connectivities_key': 'connectivities',
        'distances_key': 'distances'
    }
    
    # Add matrices to obsp
    adata.obsp['connectivities'] = connectivities
    adata.obsp['distances'] = distances
    
    # Add toy PCA and UMAP
    adata.obsm['X_pca'] = np.random.rand(n_obs, 50)  # 50 PCA components
    adata.obsm['X_umap'] = np.random.rand(n_obs, 2)  # 2D UMAP

@pytest.fixture
def random_adata_with_neighbors():
    adata = create_random_adata(n_obs=100, n_vars=3, n_clusters=5)
    add_toy_neighbors(adata)
    return adata

@pytest.fixture
def controlled_adata_with_neighbors():
    adata = create_controlled_adata()
    add_toy_neighbors(adata)
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