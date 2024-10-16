import pytest
import numpy as np
import pandas as pd

from scipy import sparse
from unittest.mock import patch

from sccellfie.preprocessing.gpr_rules import clean_gene_names, find_genes_gpr
from sccellfie.preprocessing.prepare_inputs import preprocess_inputs, stratified_subsample_adata, normalize_adata, transform_adata_gene_names
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


def test_preprocess_inputs_all_genes_half_reactions():
    # Create controlled inputs
    adata = create_controlled_adata()
    gpr_dict_expected = create_controlled_gpr_dict()
    str_gpr_dict_expected = {k: v._ast2str(v) for k, v in gpr_dict_expected.items()}
    gpr_info = pd.DataFrame.from_dict(str_gpr_dict_expected, orient='index').reset_index()
    gpr_info.columns = ['Reaction', 'GPR-symbol']
    task_by_gene = create_controlled_task_by_gene()
    rxn_by_gene = create_controlled_rxn_by_gene()
    task_by_rxn = create_controlled_task_by_rxn()

    # Preprocess inputs
    adata2, gpr_rules, task_by_gene2, rxn_by_gene2, task_by_rxn2 = preprocess_inputs(
        adata, gpr_info, task_by_gene, rxn_by_gene, task_by_rxn,
        gene_fraction_threshold=1.0, reaction_fraction_threshold=0.5
    )

    # Check if the outputs are as expected
    assert len(gpr_rules) == len(gpr_dict_expected), "Number of GPR rules should remain the same"
    assert task_by_gene2.shape[0] <= task_by_gene.shape[0], "Number of tasks should be less than or equal to the original"
    assert task_by_rxn2.shape[0] == task_by_gene2.shape[0], "Number of tasks should be consistent across task_by_gene and task_by_rxn"
    assert task_by_rxn2.shape[1] == rxn_by_gene2.shape[0], "Number of reactions should be consistent across task_by_rxn and rxn_by_gene"
    assert adata2.shape == adata.shape, "adata shape should remain the same"


def test_preprocess_inputs_half_genes_all_reactions():
    # Create controlled inputs
    adata = create_controlled_adata()
    gpr_dict_expected = create_controlled_gpr_dict()
    str_gpr_dict_expected = {k: v._ast2str(v) for k, v in gpr_dict_expected.items()}
    gpr_info = pd.DataFrame.from_dict(str_gpr_dict_expected, orient='index').reset_index()
    gpr_info.columns = ['Reaction', 'GPR-symbol']
    task_by_gene = create_controlled_task_by_gene()
    rxn_by_gene = create_controlled_rxn_by_gene()
    task_by_rxn = create_controlled_task_by_rxn()

    # Preprocess inputs
    adata2, gpr_rules, task_by_gene2, rxn_by_gene2, task_by_rxn2 = preprocess_inputs(
        adata, gpr_info, task_by_gene, rxn_by_gene, task_by_rxn,
        gene_fraction_threshold=0.5, reaction_fraction_threshold=1.0
    )

    # Check if the outputs are as expected
    assert len(gpr_rules) <= len(gpr_dict_expected), "Number of GPR rules should be less than or equal to the original"
    assert rxn_by_gene2.shape[0] <= rxn_by_gene.shape[0], "Number of reactions should be less than or equal to the original"
    assert task_by_gene2.shape[1] <= task_by_gene.shape[1], "Number of genes should be less than or equal to the original"
    assert task_by_rxn2.shape[0] == task_by_gene2.shape[0], "Number of tasks should be consistent across task_by_gene and task_by_rxn"
    assert adata2.shape[1] <= adata.shape[1], "Number of genes in adata should be less than or equal to the original"

def test_preprocess_inputs_one_gene_one_reaction():
    # Create controlled inputs
    adata = create_controlled_adata()
    gpr_dict_expected = create_controlled_gpr_dict()
    str_gpr_dict_expected = {k: v._ast2str(v) for k, v in gpr_dict_expected.items()}
    gpr_info = pd.DataFrame.from_dict(str_gpr_dict_expected, orient='index').reset_index()
    gpr_info.columns = ['Reaction', 'GPR-symbol']
    task_by_gene = create_controlled_task_by_gene()
    rxn_by_gene = create_controlled_rxn_by_gene()
    task_by_rxn = create_controlled_task_by_rxn()

    # Preprocess inputs
    adata2, gpr_rules, task_by_gene2, rxn_by_gene2, task_by_rxn2 = preprocess_inputs(
        adata, gpr_info, task_by_gene, rxn_by_gene, task_by_rxn,
        gene_fraction_threshold=0, reaction_fraction_threshold=0
    )

    # Check if the outputs are as expected
    assert len(gpr_rules) == len(gpr_dict_expected), "Number of GPR rules should remain the same"
    assert task_by_gene2.shape[0] == task_by_gene.shape[0], "Number of tasks should remain the same"
    assert rxn_by_gene2.shape[0] == rxn_by_gene.shape[0], "Number of reactions should remain the same"
    assert task_by_rxn2.shape == task_by_rxn.shape, "task_by_rxn shape should remain the same"
    assert adata2.shape == adata.shape, "adata shape should remain the same"


def test_clean_gene_names():
    test_cases = [
        ("(1 ) AND (2)", "(1) AND (2)"),
        ("( 3 ) OR ( 4 )", "(3) OR (4)"),
        ("(5) AND ( 6 ) OR (7 )", "(5) AND (6) OR (7)"),
        ("gene1 AND (8 )", "gene1 AND (8)"),
        ("( 9 ) OR gene2", "(9) OR gene2"),
        ("No parentheses here", "No parentheses here")
    ]

    for input_rule, expected_output in test_cases:
        result = clean_gene_names(input_rule)
        assert result == expected_output, f"For input '{input_rule}', expected '{expected_output}', but got '{result}'"


def test_find_genes_gpr():
    test_cases = [
        ("gene1 AND gene2", ["gene1", "gene2"]),
        ("gene3 OR (gene4 AND gene5)", ["gene3", "gene4", "gene5"]),
        ("(gene6 OR gene7) AND gene8", ["gene6", "gene7", "gene8"]),
        ("gene9 AND gene10 OR gene11", ["gene9", "gene10", "gene11"]),
        ("gene12", ["gene12"]),
        ("gene13 AND (gene14 OR (gene15 AND gene16))", ["gene13", "gene14", "gene15", "gene16"]),
        ("NO_GENES_HERE", ["NO_GENES_HERE"])
    ]

    for input_rule, expected_output in test_cases:
        result = find_genes_gpr(input_rule)
        assert result == expected_output, f"For input '{input_rule}', expected {expected_output}, but got {result}"


def test_stratified_subsample_adata():
    # Create a random AnnData object
    n_obs = 1000
    n_vars = 50
    n_clusters = 5
    adata = create_random_adata(n_obs=n_obs, n_vars=n_vars, n_clusters=n_clusters)

    # Perform stratified subsampling
    target_fraction = 0.2
    subsampled_adata = stratified_subsample_adata(adata, group_column='cluster', target_fraction=target_fraction)

    # Check if the subsampled data has approximately the correct size
    expected_size = int(n_obs * target_fraction)
    assert abs(len(subsampled_adata) - expected_size) <= n_clusters  # Allow for small rounding differences

    # Check if all clusters are represented in the subsampled data
    original_clusters = set(adata.obs['cluster'])
    subsampled_clusters = set(subsampled_adata.obs['cluster'])
    assert original_clusters == subsampled_clusters

    # Check if the proportion of each cluster is roughly maintained
    original_proportions = adata.obs['cluster'].value_counts(normalize=True)
    subsampled_proportions = subsampled_adata.obs['cluster'].value_counts(normalize=True)

    for cluster in original_clusters:
        assert np.isclose(original_proportions[cluster], subsampled_proportions[cluster], atol=0.05)


def test_normalize_adata():
    # Create controlled test data
    adata = create_controlled_adata()

    # Add total counts to the adata object
    adata.obs['n_counts'] = np.array([3, 9, 21, 21])

    # Run the preprocessing function
    adata_processed = normalize_adata(adata, target_sum=1000, n_counts_key='n_counts', copy=True)

    # Check that the data are still sparse
    assert sparse.issparse(adata_processed.X)

    # Check that the normalization was performed correctly
    expected_normalized_X = np.array([
        [333.33, 666.67, 0],
        [333.33, 444.44, 222.22],
        [238.10, 285.71, 476.19],
        [333.33, 380.95, 285.71]
    ])

    np.testing.assert_array_almost_equal(adata_processed.X.toarray(), expected_normalized_X, decimal=2)

    # Check that the normalization info was added to uns
    assert 'normalization' in adata_processed.uns
    assert adata_processed.uns['normalization']['method'] == 'total_count'
    assert adata_processed.uns['normalization']['target_sum'] == 1000
    assert adata_processed.uns['normalization']['n_counts_key'] == 'n_counts'

    # Check that the original data is preserved in .raw
    assert adata_processed.raw is not None
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    if sparse.issparse(adata_processed.raw.X):
        raw_X = adata_processed.raw.X.toarray()
    else:
        raw_X = adata_processed.raw.X
    np.testing.assert_array_equal(raw_X, X)


# Mock data for testing
MOCK_ENSEMBL2SYMBOL = {
    'ENSG00000000001': 'GENE1',
    'ENSG00000000002': 'GENE2',
    'ENSG00000000003': 'GENE3'
}


@patch('sccellfie.preprocessing.prepare_inputs.retrieve_ensembl2symbol_data')
def test_transform_adata_gene_names(mock_retrieve):
    # Mock the retrieve_ensembl2symbol_data function
    mock_retrieve.return_value = MOCK_ENSEMBL2SYMBOL

    # Load controlled adata
    controlled_adata = create_controlled_adata()
    controlled_adata.var_names = ['ENSG00000000001', 'ENSG00000000002', 'ENSG00000000003']
    original_adata = controlled_adata.copy()

    # Test with default parameters (copy=True, drop_unmapped=False)
    result = transform_adata_gene_names(controlled_adata)
    assert result is not controlled_adata  # Should be a copy
    assert list(result.var_names) == ['GENE1', 'GENE2', 'GENE3']
    assert result.X.shape == (4, 3)
    assert isinstance(result.X, sparse.csr_matrix)
    assert result.raw is not None

    # Test with copy=False
    controlled_adata = original_adata.copy()
    result = transform_adata_gene_names(controlled_adata, copy=False)
    assert result is controlled_adata  # Should be the same object
    assert list(result.var_names) == ['GENE1', 'GENE2', 'GENE3']
    assert np.array_equal(result.X.toarray(), original_adata.X.toarray())

    # Test with drop_unmapped=True (shouldn't change anything in this case)
    controlled_adata = original_adata.copy()
    result = transform_adata_gene_names(controlled_adata, drop_unmapped=True)
    assert list(result.var_names) == ['GENE1', 'GENE2', 'GENE3']
    assert result.X.shape == (4, 3)

    # Test with custom organism
    controlled_adata = original_adata.copy()
    transform_adata_gene_names(controlled_adata, organism='mouse')
    mock_retrieve.assert_called_with(None, 'mouse')

    # Test with custom filename
    controlled_adata = original_adata.copy()
    transform_adata_gene_names(controlled_adata, filename='custom.csv')
    mock_retrieve.assert_called_with('custom.csv', 'human')

    # Test error when no genes are in Ensembl format
    invalid_adata = original_adata.copy()
    invalid_adata.var_names = ['GENE1', 'GENE2', 'GENE3']
    with pytest.raises(ValueError, match="Not all genes are in Ensembl ID format"):
        transform_adata_gene_names(invalid_adata)

    # Test when retrieve_ensembl2symbol_data returns an empty dictionary
    controlled_adata = original_adata.copy()
    mock_retrieve.return_value = {}
    with pytest.raises(ValueError, match="Failed to retrieve Ensembl ID to gene symbol mapping"):
        transform_adata_gene_names(controlled_adata)

    # Test with partial mapping
    controlled_adata = original_adata.copy()
    partial_mapping = {'ENSG00000000001': 'GENE1', 'ENSG00000000002': 'GENE2'}
    mock_retrieve.return_value = partial_mapping
    result = transform_adata_gene_names(controlled_adata)
    assert list(result.var_names) == ['GENE1', 'GENE2', 'ENSG00000000003']

    # Test with partial mapping and drop_unmapped=True
    controlled_adata = original_adata.copy()
    result = transform_adata_gene_names(controlled_adata, drop_unmapped=True)
    assert list(result.var_names) == ['GENE1', 'GENE2']
    assert result.X.shape == (4, 2)