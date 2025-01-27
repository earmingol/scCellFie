import pytest
import numpy as np
import pandas as pd

from sccellfie.gene_score import compute_gene_scores, PCOUNT
from sccellfie.reaction_activity import compute_reaction_activity
from sccellfie.datasets.toy_inputs import create_controlled_adata, create_controlled_gpr_dict, create_global_threshold


@pytest.mark.parametrize("use_specificity", [True, False])
def test_compute_reaction_activity(use_specificity):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Create a small, controlled GPR dictionary
    gpr_dict = create_controlled_gpr_dict()

    # Create global threshold
    threshold_val = 0.5 # Global threshold for test
    glob_thresholds = create_global_threshold(threshold=threshold_val,
                                              n_vars=adata.shape[1])

    # Run the function
    compute_gene_scores(adata, glob_thresholds)
    compute_reaction_activity(adata,
                              gpr_dict,
                              use_specificity=use_specificity,
                              disable_pbar=True)

    # Expected reaction activity
    if use_specificity:
        denom = 2
    else:
        denom = 1
    expected_ral = np.array([[5 * np.log(1 + 1 / (threshold_val + PCOUNT)) / denom, # Cell1, Rxn1
                              5 * np.log(1 + 2 / (threshold_val + PCOUNT)), # Cell1, Rxn2
                              5 * np.log(1 + 0 / (threshold_val + PCOUNT)), # Cell1, Rxn3
                              5 * np.log(1 + 1 / (threshold_val + PCOUNT)) / denom, # Cell1, Rxn4
                              ],
                             [5 * np.log(1 + 3 / (threshold_val + PCOUNT)) / denom, # Cell2, Rxn1
                              5 * np.log(1 + 4 / (threshold_val + PCOUNT)), # Cell2, Rxn2
                              5 * np.log(1 + 2 / (threshold_val + PCOUNT)), # Cell2, Rxn3
                              5 * np.log(1 + 3 / (threshold_val + PCOUNT)) / denom, # Cell2, Rxn4
                              ],
                             [5 * np.log(1 + 5 / (threshold_val + PCOUNT)), # Cell3, Rxn1
                              5 * np.log(1 + 6 / (threshold_val + PCOUNT)) / denom, # Cell3, Rxn2
                              5 * np.log(1 + 6 / (threshold_val + PCOUNT)) / denom, # Cell3, Rxn3
                              5 * np.log(1 + 10 / (threshold_val + PCOUNT)), # Cell3, Rxn4
                              ],
                             [5 * np.log(1 + 7 / (threshold_val + PCOUNT)) / denom, # Cell4, Rxn1
                              5 * np.log(1 + 8 / (threshold_val + PCOUNT)), # Cell4, Rxn2
                              5 * np.log(1 + 6 / (threshold_val + PCOUNT)), # Cell4, Rxn3
                              5 * np.log(1 + 7 / (threshold_val + PCOUNT)) / denom, # Cell4, Rxn4
                              ],
                             ])

    # Verify the structure and values of reaction activity data
    assert hasattr(adata, 'reactions')
    np.testing.assert_almost_equal(adata.reactions.X, expected_ral, decimal=5)


def test_ral_max_genes():
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()

    # Create a small, controlled GPR dictionary
    gpr_dict = create_controlled_gpr_dict()

    # Create global threshold
    threshold_val = 0.5 # Global threshold for test
    glob_thresholds = create_global_threshold(threshold=threshold_val,
                                              n_vars=adata.shape[1])

    # Expected max genes driving RAL
    expected_max_genes = [['gene1', 'gene2', 'gene3', 'gene1'], # Cell1
                          ['gene1', 'gene2', 'gene3', 'gene1'],  # Cell2
                          ['gene1', 'gene2', 'gene2', 'gene3'],  # Cell3
                          ['gene1', 'gene2', 'gene3', 'gene1'],  # Cell4
                          ]
    exp_max_genes = pd.DataFrame(expected_max_genes,
                                 index=['cell1', 'cell2', 'cell3', 'cell4'],
                                 columns=['rxn1', 'rxn2', 'rxn3', 'rxn4']
                                 )

    # Run the function
    compute_gene_scores(adata, glob_thresholds)
    compute_reaction_activity(adata,
                              gpr_dict,
                              use_specificity=True,
                              disable_pbar=True)

    # Verify the structure and values of reaction activity data
    assert hasattr(adata, 'reactions')
    pd.testing.assert_frame_equal(adata.reactions.uns['Rxn-Max-Genes'], exp_max_genes)
