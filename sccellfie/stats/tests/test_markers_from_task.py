import pytest
import pandas as pd

from pandas.testing import assert_frame_equal
from sccellfie.stats.markers_from_task import get_task_determinant_genes
from sccellfie.gene_score import compute_gene_scores
from sccellfie.reaction_activity import compute_reaction_activity
from sccellfie.metabolic_task import compute_mt_score
from sccellfie.tests.toy_inputs import create_controlled_adata, create_controlled_task_by_rxn, create_controlled_gpr_dict, create_global_threshold


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

    # Expected determinant genes
    if groupby is None:
        expected_df = pd.DataFrame(data={'Group': ['All-Groups', 'All-Groups', 'All-Groups'],
                                         'Rxn': ['rxn3', 'rxn3', 'rxn1'],
                                         'Det-Gene': ['gene2', 'gene3', 'gene1'],
                                         'RAL': [2.740987, 2.676485, 2.585926]})
    else:
        expected_df = pd.DataFrame(data={'Group': ['B', 'B', 'B', 'A', 'A'],
                                        'Rxn': ['rxn3', 'rxn1', 'rxn3', 'rxn3', 'rxn1'],
                                        'Det-Gene': ['gene3', 'gene1', 'gene2', 'gene3', 'gene1'],
                                        'RAL': [5.481975, 3.948932, 2.740987, 1.273740, 1.22920]})
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
    expected_columns = ['Group', 'Rxn', 'Det-Gene', 'RAL']
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

    # Check if the DataFrame has the expected values
    assert_frame_equal(result_df, expected_df, check_exact=False, atol=1e-2), "Threshold values do not match expected results"



