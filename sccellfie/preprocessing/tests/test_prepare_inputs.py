import pandas as pd

from sccellfie.preprocessing import preprocess_inputs
from sccellfie.tests.toy_inputs import create_random_adata, create_controlled_adata, create_controlled_gpr_dict, create_controlled_task_by_rxn, create_controlled_task_by_gene, create_controlled_rxn_by_gene


def test_preprocess_inputs():
    # Create controlled inputs
    adata = create_controlled_adata()
    gpr_dict_expected = create_controlled_gpr_dict()
    str_gpr_dict_expected = {k : v._ast2str(v) for k, v in gpr_dict_expected.items()}
    gpr_info = pd.DataFrame.from_dict(str_gpr_dict_expected, orient='index').reset_index()
    gpr_info.columns = ['Reaction', 'GPR-symbol']
    task_by_gene = create_controlled_task_by_gene()
    rxn_by_gene = create_controlled_rxn_by_gene()
    task_by_rxn = create_controlled_task_by_rxn()

    # Preprocess inputs
    adata2, gpr_rules, task_by_gene2, rxn_by_gene2, task_by_rxn2 = preprocess_inputs(adata, gpr_info, task_by_gene, rxn_by_gene, task_by_rxn)

    # Check if the outputs are the same as the expected
    assert gpr_rules == gpr_dict_expected, "GPR rules are not the same"
    pd.testing.assert_frame_equal(task_by_gene, task_by_gene2, check_dtype=False), "task_by_gene is not the same"
    pd.testing.assert_frame_equal(rxn_by_gene, rxn_by_gene2, check_dtype=False), "rxn_by_gene is not the same"
    pd.testing.assert_frame_equal(task_by_rxn, task_by_rxn2, check_dtype=False), "task_by_rxn is not the same"


def test_shapes():
    # Create controlled inputs
    adata_controlled = create_controlled_adata()
    adata_random = create_random_adata(n_obs=10, n_vars=3)
    gpr_dict_expected = create_controlled_gpr_dict()
    str_gpr_dict_expected = {k: v._ast2str(v) for k, v in gpr_dict_expected.items()}
    gpr_info = pd.DataFrame.from_dict(str_gpr_dict_expected, orient='index').reset_index()
    gpr_info.columns = ['Reaction', 'GPR-symbol']
    task_by_gene = create_controlled_task_by_gene()
    rxn_by_gene = create_controlled_rxn_by_gene()
    task_by_rxn = create_controlled_task_by_rxn()

    # Remove one gene from adata_random that is in the controlled inputs
    adata = adata_random[:, adata_random.var_names[1:]]
    adata.raw = adata_random.raw.to_adata()[:, adata_random.var_names[1:]]
    expected_vars = adata_controlled.var_names[1:]

    # Preprocess inputs
    adata2, gpr_rules, task_by_gene2, rxn_by_gene2, task_by_rxn2 = preprocess_inputs(adata, gpr_info, task_by_gene, rxn_by_gene, task_by_rxn)

    # Check if outputs are as expected
    assert all([v in expected_vars for v in adata2.var_names]), "Not all expected genes are in adata2.var_names"
    assert all([v in expected_vars for v in task_by_gene2.columns]), "Not all expected genes are in task_by_gene2.columns"
    assert all([v in expected_vars for v in rxn_by_gene2.columns]), "Not all expected genes are in rxn_by_gene2.columns"
    assert (all([v in task_by_rxn2.index for v in task_by_gene2.index])) & (task_by_rxn2.shape[0] == task_by_gene2.shape[0]), "Tasks were not filtered correctly"
    assert (all([v in task_by_rxn2.columns for v in rxn_by_gene2.index])) & (task_by_rxn2.shape[1] == rxn_by_gene2.shape[0]), "Reactions were not filtered correctly"
    assert adata_controlled.shape != adata2.shape, "adata2 has the same shape as adata_controlled"
    assert adata_random.shape != adata2.shape, "adata2 has the same shape as adata_random"
    assert task_by_gene.shape != task_by_gene2.shape, "task_by_gene2 has the same shape as task_by_gene"
    assert rxn_by_gene.shape != rxn_by_gene2.shape, "rxn_by_gene2 has the same shape as rxn_by_gene"
    assert task_by_rxn.shape != task_by_rxn2.shape, "task_by_rxn2 has the same shape as task_by_rxn"
