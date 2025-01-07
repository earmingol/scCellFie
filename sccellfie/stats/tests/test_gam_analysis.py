import pytest
import numpy as np
import pandas as pd
from anndata import AnnData

from sccellfie.datasets.toy_inputs import create_controlled_adata, create_random_adata
from sccellfie.stats.gam_analysis import generate_pseudobulks, fit_gam_model, analyze_gam_results


def test_generate_pseudobulks():
    """Test pseudobulk generation functionality."""
    # Use random test data with more cells
    adata = create_random_adata(n_obs=100, n_vars=10, n_clusters=2)  # Use 2 clusters for simplicity
    n_pseudobulks = 2
    cells_per_bulk = 20

    # Generate pseudobulks
    adata_pseudobulk, pseudobulk_ids = generate_pseudobulks(
        adata,
        cell_type_key='cluster',
        n_pseudobulks=n_pseudobulks,
        cells_per_bulk=cells_per_bulk
    )

    # Basic checks
    assert isinstance(adata_pseudobulk, AnnData)
    assert isinstance(pseudobulk_ids, pd.Series)

    # Check number of pseudobulks (should be less than or equal to expected)
    max_expected_pseudobulks = n_pseudobulks * len(adata.obs['cluster'].unique())
    assert len(adata_pseudobulk) <= max_expected_pseudobulks

    # Check that assigned cells have valid pseudobulk IDs
    assert not pseudobulk_ids[~pseudobulk_ids.isna()].empty


def test_fit_gam_model():
    """Test GAM model fitting functionality."""
    # Create test data
    adata = create_random_adata(n_obs=50, n_vars=10, n_clusters=3)
    adata.obs['continuous_var'] = np.random.random(adata.n_obs)

    # Test with discrete categories
    gam_results = fit_gam_model(
        adata,
        cell_type_key='cluster',
        n_splines=5
    )

    # Basic checks
    assert 'models' in gam_results
    assert 'scores' in gam_results
    assert isinstance(gam_results['scores'], pd.DataFrame)
    assert len(gam_results['models']) == adata.n_vars

    # Test with continuous variable
    gam_results_continuous = fit_gam_model(
        adata,
        cell_type_key='cluster',
        continuous_key='continuous_var',
        n_splines=5
    )

    assert 'models' in gam_results_continuous
    assert len(gam_results_continuous['models']) == adata.n_vars


def test_gam_with_ordered_categories():
    """Test GAM model fitting with specific categorical order."""
    # Create test data with more observations
    adata = create_random_adata(n_obs=50, n_vars=10, n_clusters=2)

    # Define specific order for categories
    cell_type_order = ['cluster2', 'cluster1']  # Reverse the natural alphabetical order

    # Fit GAM with ordered categories
    gam_results = fit_gam_model(
        adata,
        cell_type_key='cluster',
        cell_type_order=cell_type_order,
        n_splines=5
    )

    # Verify results
    assert 'models' in gam_results
    assert 'scores' in gam_results
    assert len(gam_results['models']) == adata.n_vars

    # Verify order preservation
    first_gene = list(gam_results['models'].keys())[0]
    test_points = np.linspace(0, 1, len(cell_type_order)).reshape(-1, 1)
    predictions = gam_results['models'][first_gene].predict(test_points)
    assert len(predictions) == len(cell_type_order)


def test_analyze_gam_results():
    """Test GAM results analysis functionality."""
    # Create test data
    adata = create_random_adata(n_obs=50, n_vars=10, n_clusters=3)

    # Fit GAM models
    gam_results = fit_gam_model(
        adata,
        cell_type_key='cluster',
        n_splines=5
    )

    # Analyze results
    results_df = analyze_gam_results(
        gam_results,
        significance_threshold=0.05,
        fdr_level=0.05
    )

    # Check results
    assert isinstance(results_df, pd.DataFrame)
    assert 'adj_p_value' in results_df.columns
    assert 'significant_fdr' in results_df.columns
    assert len(results_df) == adata.n_vars


def test_pseudobulk_with_continuous():
    """Test pseudobulk generation with continuous variable aggregation."""
    # Create test data with continuous variable
    adata = create_controlled_adata()
    adata.obs['continuous_var'] = [1.0, 2.0, 3.0, 4.0]  # Add continuous variable

    # Generate pseudobulks
    adata_pseudobulk, pseudobulk_ids = generate_pseudobulks(
        adata,
        cell_type_key='group',
        n_pseudobulks=2,
        cells_per_bulk=2,
        continuous_key='continuous_var'
    )

    # Check continuous variable aggregation
    assert 'continuous_var' in adata_pseudobulk.obs.columns
    assert not adata_pseudobulk.obs['continuous_var'].isna().any()


@pytest.mark.parametrize("use_pseudobulk", [True, False])
def test_gam_model_with_options(use_pseudobulk):
    """Test GAM model fitting with different options."""
    # Create test data with more observations
    adata = create_random_adata(n_obs=100, n_vars=10, n_clusters=3)

    # Test with pseudobulk option
    gam_results = fit_gam_model(
        adata,
        cell_type_key='cluster',
        use_pseudobulk=use_pseudobulk,
        n_pseudobulks=3 if use_pseudobulk else None,
        cells_per_bulk=20 if use_pseudobulk else None
    )

    assert 'models' in gam_results
    assert 'scores' in gam_results
    if use_pseudobulk:
        assert 'pseudobulk_assignments' in gam_results
    else:
        assert 'cell_type_encoder' in gam_results