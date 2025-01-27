import numpy as np

from sccellfie.preprocessing.gpr_rules import clean_gene_names, find_genes_gpr, replace_gene_ids_in_gpr, convert_gpr_nomenclature


def test_clean_gene_names():
    test_cases = [
        ("(1 ) AND (2)", "(1) AND (2)"),
        ("( 3 ) OR ( 4 )", "(3) OR (4)"),
        ("(5) AND ( 6 ) OR (7 )", "(5) AND (6) OR (7)"),
        ("gene1 AND (8 )", "gene1 AND (8)"),
        ("( 9 ) OR gene2", "(9) OR gene2"),
        ("No parentheses here", "No parentheses here"),
        # New test cases for alphanumeric gene identifiers
        ("( GENE1 ) AND (GENE2 )", "(GENE1) AND (GENE2)"),
        ("(ABC123 ) OR ( DEF456 )", "(ABC123) OR (DEF456)"),
        ("( G1 ) AND (G2) OR ( G3 )", "(G1) AND (G2) OR (G3)"),
        ("( geneA ) AND geneB AND ( geneC )", "(geneA) AND geneB AND (geneC)"),
        ("(complex_gene_1) OR ( complex_gene_2 )", "(complex_gene_1) OR (complex_gene_2)"),
        ("( A1B2C3 ) AND (X4Y5Z6 )", "(A1B2C3) AND (X4Y5Z6)"),
        ("Mixed (GENE1) and (123 ) and ( ABC )", "Mixed (GENE1) and (123) and (ABC)"),
        ("( gene_with_underscore ) OR (geneWithCamelCase )", "(gene_with_underscore) OR (geneWithCamelCase)"),
        ("((nested_gene1)) AND (( nested_gene2 ))", "((nested_gene1)) AND ((nested_gene2))"),
        ("( UPPERCASE_GENE ) and ( lowercase_gene )", "(UPPERCASE_GENE) and (lowercase_gene)")
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


def test_replace_gene_ids_in_gpr():
    gene_id_mapping = {
        "gene1": "HGNC:1",
        "gene2": "HGNC:2",
        "gene3": "HGNC:3",
        "complex_gene": "HGNC:4"
    }

    test_cases = [
        ("gene1 AND gene2", "HGNC:1 AND HGNC:2"),
        ("(gene1) OR gene3", "HGNC:1 OR HGNC:3"),
        ("gene2 AND (gene3 OR gene1)", "HGNC:2 AND (HGNC:3 OR HGNC:1)"),
        ("(complex_gene) AND gene2", "HGNC:4 AND HGNC:2"),
        ("gene4 AND gene1", "gene4 AND HGNC:1"),  # gene4 not in mapping
        ("(gene1) AND (gene2)", "HGNC:1 AND HGNC:2")  # Test parentheses removal
    ]

    for input_rule, expected_output in test_cases:
        result = replace_gene_ids_in_gpr(input_rule, gene_id_mapping)
        assert result == expected_output, f"For input '{input_rule}', expected '{expected_output}', but got '{result}'"


def test_convert_gpr_nomenclature():
    id_mapping = {
        "gene1": "HGNC:1",
        "gene2": "HGNC:2",
        "gene3": "HGNC:3"
    }

    input_rules = [
        "gene1 AND gene2",
        "(gene1) OR gene3",
        "gene2 AND (gene3 OR gene1)",
        "gene4 AND gene1",  # gene4 not in mapping
        np.nan,  # Test handling of np.nan
        "(gene1 ) AND ( gene2)"  # Test cleaning and conversion
    ]

    expected_output = [
        "HGNC:1 AND HGNC:2",
        "HGNC:1 OR HGNC:3",
        "HGNC:2 AND (HGNC:3 OR HGNC:1)",
        "gene4 AND HGNC:1",
        np.nan,
        "HGNC:1 AND HGNC:2"
    ]

    result = convert_gpr_nomenclature(input_rules, id_mapping)

    for i, (input_rule, expected, output) in enumerate(zip(input_rules, expected_output, result)):
        if isinstance(expected, str):
            assert output == expected, f"For input '{input_rule}', expected '{expected}', but got '{output}'"
        else:
            assert np.isnan(output), f"For input '{input_rule}', expected np.nan, but got '{output}'"