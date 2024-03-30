import pytest
import numpy as np

from sccellfie.gene_score import compute_gene_scores
from sccellfie.reaction_activity import compute_reaction_activity
from sccellfie.metabolic_task import compute_mt_score
from sccellfie.tests import PCOUNT
from sccellfie.tests.toy_inputs import create_controlled_adata, create_controlled_gpr_dict, create_global_threshold, create_controlled_task_by_rxn


@pytest.mark.parametrize("use_specificity", [True, False])
def test_compute_mt_score(use_specificity):
    # Create a small, controlled AnnData object, GPR dictionary, and a task_by_rxn DataFrame
    adata = create_controlled_adata()
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
                              use_specificity=use_specificity,
                              disable_pbar=True)

    # Run the compute_mt_score function
    compute_mt_score(adata, task_by_rxn)

    # Compute the expected MTS
    if use_specificity:
        denom = 2
    else:
        denom = 1
    expected_ral = np.array([[5 * np.log(1 + 1 / (threshold_val + PCOUNT)) / denom,  # Cell1, Rxn1
                              5 * np.log(1 + 2 / (threshold_val + PCOUNT)),  # Cell1, Rxn2
                              5 * np.log(1 + 0 / (threshold_val + PCOUNT)),  # Cell1, Rxn3
                              5 * np.log(1 + 1 / (threshold_val + PCOUNT)) / denom,  # Cell1, Rxn4
                              ],
                             [5 * np.log(1 + 3 / (threshold_val + PCOUNT)) / denom,  # Cell2, Rxn1
                              5 * np.log(1 + 4 / (threshold_val + PCOUNT)),  # Cell2, Rxn2
                              5 * np.log(1 + 2 / (threshold_val + PCOUNT)),  # Cell2, Rxn3
                              5 * np.log(1 + 3 / (threshold_val + PCOUNT)) / denom,  # Cell2, Rxn4
                              ],
                             [5 * np.log(1 + 5 / (threshold_val + PCOUNT)),  # Cell3, Rxn1
                              5 * np.log(1 + 6 / (threshold_val + PCOUNT)) / denom,  # Cell3, Rxn2
                              5 * np.log(1 + 6 / (threshold_val + PCOUNT)) / denom,  # Cell3, Rxn3
                              5 * np.log(1 + 10 / (threshold_val + PCOUNT)),  # Cell3, Rxn4
                              ],
                             [5 * np.log(1 + 7 / (threshold_val + PCOUNT)) / denom,  # Cell4, Rxn1
                              5 * np.log(1 + 8 / (threshold_val + PCOUNT)),  # Cell4, Rxn2
                              5 * np.log(1 + 6 / (threshold_val + PCOUNT)),  # Cell4, Rxn3
                              5 * np.log(1 + 7 / (threshold_val + PCOUNT)) / denom,  # Cell4, Rxn4
                              ],
                             ])
    expected_mts = np.matmul(expected_ral, task_by_rxn.T) / task_by_rxn.sum(axis=1)

    # Verify the values of metabolic task scores
    assert hasattr(adata, 'metabolic_tasks')
    np.testing.assert_almost_equal(adata.metabolic_tasks.X, expected_mts, decimal=5)