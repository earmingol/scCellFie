import cobra
import pandas as pd

from sccellfie.preprocessing.gpr_rules import find_genes_gpr


def preprocess_inputs(adata, gpr_info, task_by_gene, rxn_by_gene, task_by_rxn, correction_organism='human',
                      gene_fraction_threshold=0.0, reaction_fraction_threshold=0.0, verbose=True):
    """
    Preprocesses inputs for metabolic analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    gpr_info : pandas.DataFrame
        DataFrame containing reaction IDs and their corresponding Gene-Protein-Reaction (GPR) rules.

    task_by_gene : pandas.DataFrame
        DataFrame representing the relationship between tasks and genes.

    rxn_by_gene : pandas.DataFrame
        DataFrame representing the relationship between reactions and genes.

    task_by_rxn : pandas.DataFrame
        DataFrame representing the relationship between tasks and reactions.

    correction_organism : str, optional (default: 'human')
        Organism of the input data. This is important to correct gene names that are present in
        scCellFie's or custom database. Check options in `sccellfie.preprocessing.prepare_inputs.CORRECT_GENES.keys()`

    gene_fraction_threshold : float, optional (default: 0.0)
        The minimum fraction of genes in a reaction's GPR that must be present in adata to keep the reaction.
        Range is 0 to 1.
        1.0 means all genes must be present.
        Any value > 0 and < 1 keeps reactions with at least that fraction of genes present.
        0 means keep reactions with at least one gene present.

    reaction_fraction_threshold : float, optional (default: 0.0)
        The minimum fraction of reactions in a task that must be present after gene filtering to keep the task.
        Range is 0 to 1.
        1.0 means all reactions must be present.
        Any value > 0 and < 1 keeps tasks with at least that fraction of reactions present.
        0 means keep tasks with at least one reaction present.

    verbose : bool, optional (default: True)
        If True, prints information about the preprocessing results.

    Returns
    -------
    adata2 : AnnData
        Filtered annotated data matrix.

    gpr_rules : dict
        Dictionary of GPR rules for the filtered reactions.

    task_by_gene : pandas.DataFrame
        Filtered DataFrame representing the relationship between tasks and genes.

    rxn_by_gene : pandas.DataFrame
        Filtered DataFrame representing the relationship between reactions and genes.

    task_by_rxn : pandas.DataFrame
        Filtered DataFrame representing the relationship between tasks and reactions.
    """
    if not 0 <= gene_fraction_threshold <= 1:
        raise ValueError("gene_fraction_threshold must be between 0 and 1")
    if not 0 <= reaction_fraction_threshold <= 1:
        raise ValueError("reaction_fraction_threshold must be between 0 and 1")

    adata_var = pd.DataFrame(index=adata.var.index)

    correction_col = 'corrected'
    if correction_organism in CORRECT_GENES.keys():
        correction_dict = CORRECT_GENES[correction_organism]
        correction_dict = {k : v for k, v in correction_dict.items() if v in rxn_by_gene.columns}
        adata_var[correction_col] = [correction_dict[g] if g in correction_dict.keys() else g for g in adata_var.index]
        if verbose:
            print('Gene names corrected to match database: {}'.format(len(correction_dict)))
    else:
        adata_var[correction_col] = list(adata_var.index)


    # Initialize GPRs
    gpr_rules = gpr_info.set_index('Reaction')['GPR-symbol'].to_dict()
    gpr_rules = {k: cobra.core.gene.GPR().from_string(gpr) for k, gpr in gpr_rules.items()}

    valid_genes = set()
    valid_reactions = set()

    for reaction, gpr in gpr_rules.items():
        if reaction in rxn_by_gene.index and reaction in task_by_rxn.columns:
            genes_in_rule = find_genes_gpr(gpr.to_string())
            genes_present = [gene for gene in genes_in_rule if gene in adata_var[correction_col].values]

            n_genes_in_rule = len(genes_in_rule)
            n_genes_present = len(genes_present)
            if n_genes_in_rule > 0:
                if gene_fraction_threshold == 0:
                    # Keep reaction if at least one gene is present
                    if n_genes_present > 0:
                        valid_genes.update(genes_present)
                        valid_reactions.add(reaction)
                else:
                    # Keep reaction if the fraction of present genes meets or exceeds the threshold
                    fraction_present = n_genes_present / n_genes_in_rule
                    if fraction_present >= gene_fraction_threshold:
                        valid_genes.update(genes_present)
                        valid_reactions.add(reaction)
    valid_genes = sorted(valid_genes)
    valid_reactions = sorted(valid_reactions)

    # Filter adata
    adata2 = adata[:, adata_var[correction_col].isin(valid_genes)]
    adata2.var_names = adata_var[adata_var[correction_col].isin(valid_genes)][correction_col].values.tolist()

    # Filter gene tables
    task_by_gene = task_by_gene.loc[:, valid_genes]
    rxn_by_gene = rxn_by_gene.loc[valid_reactions, valid_genes]

    # Filter tasks based on reaction presence
    valid_tasks = set()
    for task in task_by_rxn.index:
        rxns_in_task = task_by_rxn.loc[task]
        rxns_present = rxns_in_task[rxns_in_task.index.isin(valid_reactions)]

        n_rxns_in_task = rxns_in_task.sum()
        n_rxns_present = rxns_present.sum()
        if n_rxns_in_task > 0:
            if reaction_fraction_threshold == 0:
                # Keep task if at least one reaction is present
                if n_rxns_present > 0:
                    valid_tasks.add(task)
            else:
                # Keep task if the fraction of present reactions meets or exceeds the threshold
                fraction_present = n_rxns_present / n_rxns_in_task
                if fraction_present >= reaction_fraction_threshold:
                    valid_tasks.add(task)
    valid_tasks = sorted(valid_tasks)

    # Final filtering of task tables
    task_by_gene = task_by_gene.loc[valid_tasks]
    task_by_rxn = task_by_rxn.loc[valid_tasks, valid_reactions]

    # Remove genes and reactions with no non-zero values
    task_by_gene = task_by_gene.loc[:, (task_by_gene != 0).any(axis=0)]
    rxn_by_gene = rxn_by_gene.loc[(rxn_by_gene != 0).any(axis=1), (rxn_by_gene != 0).any(axis=0)]
    task_by_rxn = task_by_rxn.loc[:, (task_by_rxn != 0).any(axis=0)]

    # Update valid genes and reactions
    valid_genes = sorted(set(task_by_gene.columns))
    valid_reactions = sorted(set(task_by_rxn.columns))

    # Update GPR rules
    gpr_rules = {k: v for k, v in gpr_rules.items() if k in valid_reactions}

    # Final update
    rxn_by_gene = rxn_by_gene.loc[valid_reactions, valid_genes]
    task_by_gene = task_by_gene.loc[valid_tasks, valid_genes]
    task_by_rxn = task_by_rxn.loc[valid_tasks, valid_reactions]
    adata2 = adata2[:, valid_genes]

    if verbose:
        print(f'Shape of new adata object: {adata2.shape}\n'
              f'Number of GPRs: {len(gpr_rules)}\n'
              f'Shape of tasks by genes: {task_by_gene.shape}\n'
              f'Shape of reactions by genes: {rxn_by_gene.shape}\n'
              f'Shape of tasks by reactions: {task_by_rxn.shape}')

    return adata2, gpr_rules, task_by_gene, rxn_by_gene, task_by_rxn


# Gene name in dataset to gene name in scCellFie's DB.
CORRECT_GENES = {'human' : {'ADSS': 'ADSS2',
                            'ADSSL1': 'ADSS1',
                            'COL4A3BP': 'CERT1',
                            'MT-CO1': 'COX1',
                            'MT-CO2': 'COX2',
                            'MT-CO3': 'COX3',
                            'MT-CYB': 'CYTB',
                            'ATP5S': 'DMAC2L',
                            'FUK': 'FCSK',
                            'G6PC': 'G6PC1',
                            'WRB': 'GET1',
                            'ASNA1': 'GET3',
                            'MARCH6': 'MARCHF6',
                            'MUT': 'MMUT',
                            'MT-ND1': 'ND1',
                            'MT-ND2': 'ND2',
                            'MT-ND3': 'ND3',
                            'MT-ND4': 'ND4',
                            'MT-ND4L': 'ND4L',
                            'MT-ND5': 'ND5',
                            'MT-ND6': 'ND6',
                            'ZADH2': 'PTGR3',
                            },
                 'mouse' : {'Gars1': 'Gars',
                            'Srpr' : 'Srpra',
                            'mt-Cytb' : 'Cytb',
                            'mt-Nd1' : 'Nd1',
                            'mt-Nd2' : 'Nd2',
                            'mt-Nd3' : 'Nd3',
                            'mt-Nd4' : 'Nd4',
                            'mt-Nd4l' : 'Nd4l',
                            'mt-Nd5' : 'Nd5',
                            'mt-Nd6' : 'Nd6',
                            'Sdr42e2' : 'Gm5737',
                            'Klk1b26' : 'Egfbp2',
                            'Il4i1' : 'Il4i1b',
                            }
                 }