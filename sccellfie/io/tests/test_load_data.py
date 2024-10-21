import pytest
import os
import tempfile

import scanpy as sc
import networkx as nx

from sccellfie.io import load_adata, save_adata
from sccellfie.gene_score import compute_gene_scores
from sccellfie.metabolic_task import compute_mt_score
from sccellfie.reaction_activity import compute_reaction_activity
from sccellfie.spatial import create_knn_network
from sccellfie.datasets.toy_inputs import create_random_adata, create_controlled_adata_with_spatial, create_controlled_gpr_dict, create_global_threshold, create_controlled_task_by_rxn


@pytest.mark.parametrize("reactions_filename, mt_filename", [
    (None, None),
    ('custom_rxn', None),
    (None, 'custom_tasks'),
    ('custom_rxn', 'custom_tasks')
])
def test_load_all_components(reactions_filename, mt_filename):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Setup - Create data with all components
        # Create a small, controlled AnnData object, GPR dictionary, and a task_by_rxn DataFrame
        adata = create_controlled_adata_with_spatial()
        gpr_dict = create_controlled_gpr_dict()
        task_by_rxn = create_controlled_task_by_rxn()

        # Set up thresholds
        threshold_val = 0.5  # Global threshold for test
        glob_thresholds = create_global_threshold(threshold=threshold_val,
                                                  n_vars=adata.shape[1])

        # Run the reaction activity computation
        compute_gene_scores(adata, glob_thresholds)
        compute_reaction_activity(adata,
                                  gpr_dict,
                                  use_specificity=True,
                                  disable_pbar=True)

        # Run the compute_mt_score function
        compute_mt_score(adata, task_by_rxn)

        # Create KNN network
        create_knn_network(adata, n_neighbors=2, spatial_key='X_spatial', added_key='spatial_network')
        create_knn_network(adata.reactions, n_neighbors=2, spatial_key='X_spatial', added_key='spatial_network')
        create_knn_network(adata.metabolic_tasks, n_neighbors=2, spatial_key='X_spatial', added_key='spatial_network')

        # Save
        save_adata(adata, tmpdirname, filename='test_data')

        if reactions_filename is not None:
            if os.path.exists(f"{tmpdirname}/test_data_reactions.h5ad"):
                os.rename(f"{tmpdirname}/test_data_reactions.h5ad", f"{tmpdirname}/{reactions_filename}.h5ad")
        if mt_filename is not None:
            if os.path.exists(f"{tmpdirname}/test_data_metabolic_tasks.h5ad"):
                os.rename(f"{tmpdirname}/test_data_metabolic_tasks.h5ad", f"{tmpdirname}/{mt_filename}.h5ad")

        # Load
        loaded_adata = load_adata(tmpdirname, 'test_data', reactions_filename, mt_filename)

        # Assertions
        assert isinstance(loaded_adata, sc.AnnData)
        assert hasattr(loaded_adata, 'reactions')
        assert hasattr(loaded_adata, 'metabolic_tasks')
        assert 'gene_scores' in loaded_adata.layers.keys()
        assert 'Rxn-Max-Genes' in loaded_adata.reactions.uns.keys()

        # Other attributes
        assert isinstance(loaded_adata.uns['spatial_network']['graph'], nx.Graph)
        assert isinstance(loaded_adata.reactions.uns['spatial_network']['graph'], nx.Graph)
        assert isinstance(loaded_adata.metabolic_tasks.uns['spatial_network']['graph'], nx.Graph)

def test_load_without_optional_files():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create sample AnnData
        adata = create_random_adata(n_obs=20, n_vars=10)

        # Add reactions and metabolic tasks objects
        adata.reactions = create_random_adata(n_obs=20, n_vars=10)
        adata.metabolic_tasks = create_random_adata(n_obs=20, n_vars=10)

        # Call your save function
        save_adata(adata, tmpdirname, 'test_data')

        if os.path.exists(f"{tmpdirname}/test_data_reactions.h5ad"):
            os.rename(f"{tmpdirname}/test_data_reactions.h5ad", f"{tmpdirname}/test_data_reactions2.h5ad")

        if os.path.exists(f"{tmpdirname}/test_data_metabolic_tasks.h5ad"):
            os.rename(f"{tmpdirname}/test_data_metabolic_tasks.h5ad", f"{tmpdirname}/test_data_metabolic_tasks2.h5ad")

        loaded_adata = load_adata(tmpdirname, 'test_data')

        # Assertions
        assert isinstance(loaded_adata, sc.AnnData)
        assert not hasattr(loaded_adata, 'reactions')
        assert not hasattr(loaded_adata, 'metabolic_tasks')