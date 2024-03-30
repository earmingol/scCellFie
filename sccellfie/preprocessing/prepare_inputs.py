import cobra


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
    gpr_rules = gpr_info.dropna().set_index('Reaction').to_dict()['GPR-symbol']

    genes = [g for g in rxn_by_gene.columns if (g in task_by_gene.columns) & (g in adata.var_names)]

    task_by_gene = task_by_gene.loc[:, genes]
    task_by_gene = task_by_gene.loc[(task_by_gene != 0).any(axis=1)]

    rxn_by_gene = rxn_by_gene.loc[:, genes]
    rxn_by_gene = rxn_by_gene.loc[(rxn_by_gene != 0).any(axis=1)]

    gpr_rules = {k: v.replace(' AND ', ' and ').replace(' OR ', ' or ') for k, v in gpr_rules.items() if k in rxn_by_gene.index.tolist()}

    # Initialize GPRs
    gpr_rules = {k: cobra.core.gene.GPR().from_string(gpr) for k, gpr in gpr_rules.items()}

    tasks = task_by_gene.index.tolist()
    rxns = list(gpr_rules.keys())

    task_by_rxn = task_by_rxn.loc[tasks, rxns]
    task_by_rxn = task_by_rxn.loc[(task_by_rxn != 0).any(axis=1)]

    adata2 = adata[:, [True if g in genes else False for g in adata.var_names]]
    if hasattr(adata, 'raw'):
        if adata.raw is not None:
            adata2.raw = adata.raw.to_adata()[:, [True if g in genes else False for g in adata.var_names]]

    if verbose:
        print(f'Shape of new adata object: {adata2.shape}\n'
              f'Number of GPRs: {len(gpr_rules)}\n'
              f'Shape of tasks by genes: {task_by_gene.shape}\n'
              f'Shape of reactions by genes: {rxn_by_gene.shape}\n'
              f'Shape of tasks by reactions: {task_by_rxn.shape}')

    return adata2, gpr_rules, task_by_gene, rxn_by_gene, task_by_rxn