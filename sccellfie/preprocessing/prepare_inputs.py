import cobra
import re


def preprocess_inputs(adata, gpr_info, task_by_gene, rxn_by_gene, task_by_rxn, verbose=True):
    '''
    Preprocess inputs for sccellfie.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    gpr_info: pandas.DataFrame
        A pandas.DataFrame object with the GPR information for each reaction.

    task_by_gene: pandas.DataFrame
        A pandas.DataFrame object where rows are metabolic tasks and columns are
        genes. Each cell contains ones or zeros, indicating whether a gene
        is involved in a metabolic task.

    rxn_by_gene: pandas.DataFrame
        A pandas.DataFrame object where rows are reactions and columns are
        genes. Each cell contains ones or zeros, indicating whether a gene
        is involved in a reaction.

    task_by_rxn: pandas.DataFrame
        A pandas.DataFrame object where rows are metabolic tasks and columns are
        reactions. Each cell contains ones or zeros, indicating whether a reaction
        is involved in a metabolic task.

    verbose: bool, optional (default: True)
        Whether to print information about the preprocessing.

    Returns
    -------
    adata2: AnnData object
        Annotated data matrix with only genes present in task_by_gene, rxn_by_gene, and adata.var_names.

    gpr_rules: dict
        A dictionary with reaction IDs as keys and Gene-Protein Rules (GPRs) as values.
        GPRs are in the format of the ast library after initialization with cobra from strings
        to tree format for AND and OR rules.

    task_by_gene: pandas.DataFrame
        A pandas.DataFrame object where rows are metabolic tasks and columns are
        genes. Each cell contains ones or zeros, indicating whether a gene
        is involved in a metabolic task.

    rxn_by_gene: pandas.DataFrame
        A pandas.DataFrame object where rows are reactions and columns are
        genes. Each cell contains ones or zeros, indicating whether a gene
        is involved in a reaction.

    task_by_rxn: pandas.DataFrame
        A pandas.DataFrame object where rows are metabolic tasks and columns are
        reactions. Each cell contains ones or zeros, indicating whether a reaction
        is involved in a metabolic task.
    '''
    # Initialize GPRs
    gpr_rules = gpr_info.dropna().set_index('Reaction').to_dict()['GPR-symbol']

    valid_genes = set()
    valid_reactions = set()

    # Iterate through GPR rules
    for reaction, gpr_rule in gpr_rules.items():
        if reaction in rxn_by_gene.index and reaction in task_by_rxn.columns:
            clean_gpr_rule = clean_gene_names(gpr_rule)
            genes_in_rule = find_genes_gpr(clean_gpr_rule)
            if all(gene in adata.var_names for gene in genes_in_rule):
                valid_genes.update(genes_in_rule)
                valid_reactions.add(reaction)

    # Filter adata
    adata2 = adata[:, sorted(valid_genes)]

    # Filter tables
    task_by_gene = task_by_gene.loc[:, sorted(valid_genes)]
    rxn_by_gene = rxn_by_gene.loc[sorted(valid_reactions), sorted(valid_genes)]
    task_by_rxn = task_by_rxn.loc[:, sorted(valid_reactions)]

    # Remove tasks with no non-zero values
    task_by_gene = task_by_gene.loc[(task_by_gene != 0).any(axis=1)]
    task_by_rxn = task_by_rxn.loc[(task_by_rxn != 0).any(axis=1)]

    # Ensure consistency across all tables
    common_tasks = set(task_by_gene.index) & set(task_by_rxn.index)
    task_by_gene = task_by_gene.loc[sorted(common_tasks)]
    task_by_rxn = task_by_rxn.loc[sorted(common_tasks)]

    # Update GPR rules
    gpr_rules = {k: v for k, v in gpr_rules.items() if k in valid_reactions}

    # Initialize GPRs
    gpr_rules = {k: cobra.core.gene.GPR().from_string(gpr) for k, gpr in gpr_rules.items()}

    if verbose:
        print(f'Shape of new adata object: {adata2.shape}\n'
              f'Number of GPRs: {len(gpr_rules)}\n'
              f'Shape of tasks by genes: {task_by_gene.shape}\n'
              f'Shape of reactions by genes: {rxn_by_gene.shape}\n'
              f'Shape of tasks by reactions: {task_by_rxn.shape}')

    return adata2, gpr_rules, task_by_gene, rxn_by_gene, task_by_rxn


def clean_gene_names(gpr_rule):
    # Regular expression pattern to match spaces between numbers and parentheses
    pattern = r'(\()\s*(\d+)\s*(\))'
    # Replace the matched pattern with parentheses directly around the numbers
    cleaned_rule = re.sub(pattern, r'(\2)', gpr_rule)
    return cleaned_rule


def find_genes_gpr(gpr_rule):
    elements = re.findall(r'\b[^\s(),]+\b', gpr_rule)
    return [e for e in elements if e.lower() not in ('and', 'or')]