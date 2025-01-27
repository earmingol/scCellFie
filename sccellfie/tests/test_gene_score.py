import pytest

import numpy as np
import pandas as pd

from cobra.core.gene import GPR
from scipy.sparse import issparse

from sccellfie.gene_score import gene_score, compute_gene_scores, compute_gpr_gene_score, PCOUNT
from sccellfie.datasets.toy_inputs import create_controlled_adata


def test_gene_score():
    # Test data
    gene_expression = np.array([1, 2, 3])
    gene_threshold = np.array([0.1, 0.2, 0.3])

    # Expected results
    expected_scores = 5 * np.log(1 + gene_expression / (gene_threshold + PCOUNT))

    # Actual results
    actual_scores = gene_score(gene_expression, gene_threshold)

    # Assert equality
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-5)


@pytest.mark.parametrize("use_raw", [False, True])
def test_compute_gene_scores(use_raw):
    # Create a small, controlled AnnData object
    adata = create_controlled_adata()
    if use_raw:
        X = adata.raw.X
    else:
        X = adata.X

    if issparse(X):
        X = X.toarray()

    # Define known thresholds
    thresholds = pd.DataFrame({'gene_threshold': [0.5, 3, 5]}, index=['gene1', 'gene2', 'gene3'])

    # Expected gene scores based on the defined formula
    expected_scores = 5 * np.log(1 + X / (thresholds.values.T + PCOUNT))

    # Compute gene scores using the function
    compute_gene_scores(adata, thresholds, use_raw=use_raw)

    # Retrieve the computed gene scores from adata
    computed_scores = adata.layers['gene_scores']

    # Check if computed scores match the expected values
    np.testing.assert_allclose(computed_scores, expected_scores, rtol=1e-5)



gene_values = {'gene1': 1, 'gene2': 2, 'gene3': 5, 'gene4': 0, 'gene5': 10, 'gene6': 7}

# Helper function to convert GPR string to AST structure
def gpr_to_ast(gpr_string):
    # Parse the GPR string into an AST format expected by evaluate_gene_score
    gpr_ast = GPR().from_string(gpr_string)
    return gpr_ast

@pytest.mark.parametrize("gpr_string,expected_score", [
    ("gene1 and gene2", 1.0),
    ("gene1 or gene2", 2.0),
    ("(gene1 and gene2) or gene3", 5.0),
    ("gene1 and (gene2 or gene3)", 1.0),
    ("(gene1 or gene2) and (gene3 or gene4)", 2.0),
    ("(gene1 and gene2 and gene3) or (gene4 and gene5)", 1.0),
    ("(gene1 or gene2 or gene3) and (gene4 or gene5 or gene6)", 5.0),
    ("(gene1 or gene2 or gene3) or (gene4 or gene5 or gene6)", 10.0),
    ("(gene1 or gene2 or gene3) and (gene4 or gene5 or gene6) and geneX", 0.0),
])
def test_evaluate_gene_score(gpr_string, expected_score):
    gpr_ast = gpr_to_ast(gpr_string)
    score, _ = compute_gpr_gene_score(gpr_ast, gene_values)
    assert score == expected_score, f"For GPR '{gpr_string}', expected {expected_score}, got {score}"


@pytest.mark.parametrize("gpr_string,expected_gene", [
    ("gene1 and gene2", 'gene1'),
    ("gene1 or gene2", 'gene2'),
    ("(gene1 and gene2) or gene3", 'gene3'),
    ("gene1 and (gene2 or gene3)", 'gene1'),
    ("(gene1 or gene2) and (gene3 or gene4)", 'gene2'),
    ("(gene1 and gene2 and gene3) or (gene4 and gene5)", 'gene1'),
    ("(gene1 or gene2 or gene3) and (gene4 or gene5 or gene6)", 'gene3'),
    ("(gene1 or gene2 or gene3) or (gene4 or gene5 or gene6)", 'gene5'),
    ("(gene1 or gene2 or gene3) and (gene4 or gene5 or gene6) and geneX", 'geneX'),
])
def test_evaluate_determinant_gene(gpr_string, expected_gene):
    gpr_ast = gpr_to_ast(gpr_string)
    _, gene = compute_gpr_gene_score(gpr_ast, gene_values)
    assert gene == expected_gene, f"For GPR '{gpr_string}', expected {expected_gene}, got {gene}"