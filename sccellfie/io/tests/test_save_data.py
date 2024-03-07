import pytest
import os
import tempfile

from sccellfie.io.save_data import save_adata
from sccellfie.spatial import create_knn_network
from sccellfie.tests.toy_inputs import create_random_adata, create_random_adata_with_spatial


@pytest.mark.parametrize("spatial_data, test_attr", [(True, False), (False, False), (False, True)]) # Test with and without spatial data
def test_save_adata(spatial_data, test_attr):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create sample AnnData
        if spatial_data:
            adata = create_random_adata_with_spatial(n_obs=20, n_vars=10, spatial_key='X_spatial')
        else:
            adata = create_random_adata(n_obs=20, n_vars=10)

        # Add reactions and metabolic tasks objects
        adata.reactions = create_random_adata_with_spatial(n_obs=20, n_vars=10, spatial_key='X_spatial') if spatial_data else create_random_adata(n_obs=20, n_vars=10)
        adata.metabolic_tasks = create_random_adata_with_spatial(n_obs=20, n_vars=10, spatial_key='X_spatial') if spatial_data else create_random_adata(n_obs=20, n_vars=10)

        # Generate spatial network to test storage of networkx object as pandas dataframe
        if spatial_data:
            create_knn_network(adata, n_neighbors=5, spatial_key='X_spatial', added_key='spatial_network')
            create_knn_network(adata.reactions, n_neighbors=5, spatial_key='X_spatial', added_key='spatial_network')
            create_knn_network(adata.metabolic_tasks, n_neighbors=5, spatial_key='X_spatial', added_key='spatial_network')

        if test_attr:
            delattr(adata, 'reactions')
            delattr(adata, 'metabolic_tasks')

        # Call your save function
        save_adata(adata, tmpdirname, 'test_data')

        # Assertions
        assert os.path.exists(os.path.join(tmpdirname, 'test_data.h5ad'))
        if not test_attr:
            assert os.path.exists(os.path.join(tmpdirname, 'test_data_reactions.h5ad'))
            assert os.path.exists(os.path.join(tmpdirname, 'test_data_metabolic_tasks.h5ad'))
