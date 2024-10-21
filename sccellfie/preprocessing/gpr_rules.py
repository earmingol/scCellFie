import re
import numpy as np


def clean_gene_names(gpr_rule):
    """
    Removes spaces between parentheses and gene IDs in a GPR rule.

    Parameters
    ----------
    gpr_rule : str
        GPR rule to clean.

    Returns
    -------
    cleaned_gpr : str
        Cleaned GPR rule, without spaces between parentheses and gene IDs.
    """
    # Remove spaces between parentheses and gene IDs
    cleaned_gpr = re.sub(r'\(\s*(\w+)\s*\)', r'(\1)', gpr_rule)
    return cleaned_gpr


def find_genes_gpr(gpr_rule):
    """
    Finds all gene IDs in a GPR rule.

    Parameters
    ----------
    gpr_rule : str
        GPR rule to search for gene IDs.

    Returns
    -------
    genes : list of str
        List of gene IDs found in the GPR rule.
    """
    elements = re.findall(r'\b[^\s(),]+\b', gpr_rule)
    genes = [e for e in elements if e.lower() not in ('and', 'or')]
    return genes


def replace_gene_ids_in_gpr(gpr_rule, gene_id_mapping):
    """
    Replaces gene IDs in a GPR rule with new IDs (different nomenclature).

    Parameters
    ----------
    gpr_rule : str
        GPR rule to update.

    gene_id_mapping : dict
        Dictionary mapping old gene IDs to new gene IDs.

    Returns
    -------
    updated_gpr_rule : str
        GPR rule with gene IDs replaced by new IDs.
    """
    updated_gpr_rule = gpr_rule
    for gene_id, new_id in gene_id_mapping.items():
        # Replace gene_id when it's surrounded by parentheses, removing the parentheses
        updated_gpr_rule = re.sub(rf'\({re.escape(gene_id)}\)', new_id, updated_gpr_rule)
        # Replace gene_id when it's not surrounded by parentheses
        updated_gpr_rule = re.sub(rf'\b{re.escape(gene_id)}\b', new_id, updated_gpr_rule)
    return updated_gpr_rule


def convert_gpr_nomenclature(gpr_rules, id_mapping):
    """
    Converts gene IDs in multiple GPR rules to a different nomenclature.

    Parameters
    ----------
    gpr_rules : list of str
        List of GPR rules to update.

    id_mapping : dict
        Dictionary mapping old gene IDs to new gene IDs.

    Returns
    -------
    converted_rules : list of str
        List of GPR rules with gene IDs replaced by new IDs.
    """
    converted_rules = []
    for gpr in gpr_rules:
        if isinstance(gpr, str):
            cleaned_gpr = clean_gene_names(gpr)
            converted_gpr = replace_gene_ids_in_gpr(cleaned_gpr, id_mapping)
            converted_rules.append(converted_gpr)
        else:
            converted_rules.append(np.nan)
    return converted_rules