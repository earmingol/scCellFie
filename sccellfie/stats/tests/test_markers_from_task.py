import pytest
import pandas as pd

from pandas.testing import assert_frame_equal
from sccellfie.stats.markers_from_task import get_task_determinant_genes
from sccellfie.gene_score import compute_gene_scores
from sccellfie.reaction_activity import compute_reaction_activity
from sccellfie.metabolic_task import compute_mt_score
from sccellfie.datasets.toy_inputs import create_controlled_adata, create_controlled_task_by_rxn, create_controlled_gpr_dict, create_global_threshold


@pytest.mark.parametrize("groupby, group, min_activity",
                         [(None, None, 0.0),
                          ('group', None, 0.0),
                          ('group', 'A', 0.0),
                          ('group', 'B', 0.0),
                          ('group', ['A', 'B'], 0.0),
                          (None, None, 2.5),
                          ('group', None, 0.0),
                          ('group', 'A', 1.25),
                          ('group', 'B', 3.0),
                          (None, 'B', 0.0),
                         ])
def test_get_task_determinant_genes(groupby, group, min_activity):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Create a small, controlled GPR dictionary
    gpr_dict = create_controlled_gpr_dict()

    # Create a small, controlled task-by-reaction DataFrame
    task_by_rxn = create_controlled_task_by_rxn()

    # Create global threshold
    threshold_val = 3  # Global threshold for test
    glob_thresholds = create_global_threshold(threshold=threshold_val,
                                              n_vars=adata.shape[1])

    # Run the function
    compute_gene_scores(adata, glob_thresholds)
    compute_reaction_activity(adata,
                              gpr_dict,
                              disable_pbar=True)
    compute_mt_score(adata, task_by_rxn)

    result_df = get_task_determinant_genes(adata,
                                           metabolic_task='task1',
                                           task_by_rxn=task_by_rxn,
                                           groupby=groupby,
                                           group=group,
                                           min_activity=min_activity)

    # Expected determinant genes with cell fractions
    if groupby is None:
        # All cells are treated as one group
        total_cells = adata.n_obs
        expected_df = pd.DataFrame(data={
            'Group': ['All-Groups', 'All-Groups', 'All-Groups'],
            'Rxn': ['rxn3', 'rxn3', 'rxn1'],
            'Det-Gene': ['gene2', 'gene3', 'gene1'],
            'RAL': [2.746531, 2.682397, 2.591538],
            'Cell_fraction': [0.25, 0.75, 1.0]  # For rxn3: gene2 in 1/4 cells, gene3 in 3/4 cells. For rxn1: gene1 in all cells
        })
    else:
        # Split by groups A and B
        expected_df = pd.DataFrame(data={
            'Group': ['B', 'B', 'B', 'A', 'A'],
            'Rxn': ['rxn3', 'rxn1', 'rxn3', 'rxn3', 'rxn1'],
            'Det-Gene': ['gene3', 'gene1', 'gene2', 'gene3', 'gene1'],
            'RAL': [5.493061, 3.957039, 2.746531, 1.277064, 1.226037],
            'Cell_fraction': [0.5, 1.0, 0.5, 1.0, 1.0]  # Group B: gene3 in 1/2 cells and gene2 in 1/2 cells for rxn3, gene1 in 2/2 for rxn1; Group A: gene3 in 2/2 cells for rxn3, gene1 in 2/2 for rxn1
        })
        if group is not None:
            if isinstance(group, list):
                groups = group
            else:
                groups = [group]
            expected_df = expected_df[expected_df['Group'].isin(groups)].reset_index(drop=True)

    expected_df = expected_df[expected_df['RAL'] >= min_activity].reset_index(drop=True)

    # Check if the result is a DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = ['Group', 'Rxn', 'Det-Gene', 'RAL', 'Cell_fraction']
    assert all(column in result_df.columns for column in expected_columns)

    # Check if the DataFrame filters by min_activity correctly
    if min_activity != 0.:
        assert all(result_df['RAL'] >= min_activity)

    # Check if the DataFrame contains the expected groups
    if (groupby is not None) & (group is not None):
        if isinstance(group, list):
            groups = group
        else:
            groups = [group]
        assert set(result_df['Group'].unique()) == set(groups)

    # Check Cell_fraction values
    assert all(0 <= frac <= 1 for frac in result_df['Cell_fraction']), "Cell fractions should be between 0 and 1"

    # Check if Cell_fractions sum to 1 within each Group-Rxn combination
    # Only check sum=1 constraint when no activity filtering is applied
    if min_activity == 0.0:
        group_rxn_sums = result_df.groupby(['Group', 'Rxn'])['Cell_fraction'].sum()
        assert all(abs(sum - 1.0) < 1e-6 for sum in
                   group_rxn_sums), "Cell fractions should sum to 1 for each Group-Rxn combination"

    # Check if the DataFrame has the expected values
    assert_frame_equal(result_df, expected_df, check_exact=False, atol=1e-2), "Values do not match expected results"