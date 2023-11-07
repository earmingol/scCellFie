import re


def replace_gene_ids_in_gpr(gpr_rule, gene_id_mapping):
    # Use regular expressions to find gene IDs in the GPR rule
    elements = re.findall(r'\b[\w:]+\b', gpr_rule)
    gene_ids = [e for e in elements if e not in ('and', 'or', 'AND', 'OR')]

    # Replace gene IDs with their mapped values
    updated_gpr_rule = gpr_rule
    for gene_id in gene_ids:
        if gene_id in gene_id_mapping:
            updated_gpr_rule = updated_gpr_rule.replace(gene_id, gene_id_mapping[gene_id])

    return updated_gpr_rule