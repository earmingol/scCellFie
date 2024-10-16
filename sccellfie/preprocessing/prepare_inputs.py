import cobra
import warnings
import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse

from sccellfie.datasets.gene_info import retrieve_ensembl2symbol_data
from sccellfie.preprocessing.gpr_rules import find_genes_gpr


def preprocess_inputs(adata, gpr_info, task_by_gene, rxn_by_gene, task_by_rxn, correction_organism='human',
                      gene_fraction_threshold=0.0, reaction_fraction_threshold=0.0, verbose=True):
    """
    Preprocesses inputs for metabolic analysis.

    Parameters:
    -----------
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

    correction_organism : str, optional (default='human')
        Organism of the input data. This is important to correct gene names that are present in
        scCellFie's or custom database. Check options in `sccellfie.preprocessing.prepare_inputs.CORRECT_GENES.keys()`

    gene_fraction_threshold : float, optional (default=0.0)
        The minimum fraction of genes in a reaction's GPR that must be present in adata to keep the reaction.
        Range is 0 to 1.
        1.0 means all genes must be present.
        Any value > 0 and < 1 keeps reactions with at least that fraction of genes present.
        0 means keep reactions with at least one gene present.

    reaction_fraction_threshold : float, optional (default=0.0)
        The minimum fraction of reactions in a task that must be present after gene filtering to keep the task.
        Range is 0 to 1.
        1.0 means all reactions must be present.
        Any value > 0 and < 1 keeps tasks with at least that fraction of reactions present.
        0 means keep tasks with at least one reaction present.

    verbose : bool, optional (default=True)
        If True, prints information about the preprocessing results.

    Returns:
    --------
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


def stratified_subsample_adata(adata, group_column, target_fraction=0.20, random_state=0):
    """
    Stratified subsampling of an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    group_column : str
        Column name in adata.obs containing the group information.

    target_fraction : float, optional (default=0.20)
        Fraction of cells to sample from each group.

    random_state : int, optional (default=0)
        Random seed for reproducibility.

    Returns
    -------
    adata_subsampled : AnnData
        Subsampled AnnData object
    """
    np.random.seed(random_state)

    # Get the group categories
    categories = adata.obs[group_column].cat.categories

    # Initialize an empty list to store subsampled indices
    subsampled_indices = []

    # Perform stratified subsampling
    for category in categories:
        # Get indices for the current category
        category_indices = adata.obs[adata.obs[group_column] == category].index

        # Calculate the number of cells to sample from this category
        n_sample = int(len(category_indices) * target_fraction)

        # Randomly sample indices
        sampled_indices = np.random.choice(category_indices, size=n_sample, replace=False)

        # Add sampled indices to the list
        subsampled_indices.extend(sampled_indices)

    # Convert the list of indices to a pandas Index
    subsampled_indices = pd.Index(subsampled_indices)

    # Return the subsampled AnnData object
    adata_subsampled = adata[subsampled_indices]
    return adata_subsampled


def normalize_adata(adata, target_sum=10_000, n_counts_key='n_counts', copy=False):
    """
    Preprocesses an AnnData object by normalizing the data to a target sum.
    Original adata object is updated in place.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the expression data.

    target_sum : int, optional (default=10_000)
        The target sum to which the data will be normalized.

    n_counts_key : str, optional (default='n_counts')
        The key in adata.obs containing the total counts for each cell.

    copy : bool, optional (default=False)
        If True, returns a copy of adata with the normalized data.
    """
    if copy:
        adata = adata.copy()

    # Check if total counts are already calculated
    if n_counts_key not in adata.obs.columns:
        warnings.warn(f"{n_counts_key} not found in adata.obs. Calculating total counts.", UserWarning)
        sc.pp.calculate_qc_metrics(adata, layer=None, inplace=True)
        n_counts_key = 'total_counts'  # scanpy uses 'total_counts' as the key

    # Input data
    X_view = adata.X

    warnings.warn("Normalizing data.", UserWarning)

    # Check if matrix is sparse
    is_sparse = sparse.issparse(X_view)

    # Convert to dense if sparse
    if is_sparse:
        X_view = X_view.toarray()

    # Normalize
    n_counts = adata.obs[n_counts_key].values[:, None]
    X_norm = X_view / n_counts * target_sum

    # Convert back to sparse if original was sparse
    if is_sparse:
        X_norm = sparse.csr_matrix(X_norm)

    # Update adata
    adata.X = X_norm
    adata.uns['normalization'] = {
        'method': 'total_count',
        'target_sum': target_sum,
        'n_counts_key': n_counts_key
    }
    if copy:
        return adata


def transform_adata_gene_names(adata, filename=None, organism='human', copy=True, drop_unmapped=False):
    """
    Transforms gene names in an AnnData object from Ensembl IDs to gene symbols.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the expression data. All gene names
        must be in Ensembl ID format.

    filename : str, optional
        The file path to a custom CSV file containing Ensembl IDs and gene symbols.
        One column must be 'ensembl_id' and the other 'symbol'.

    organism : str, optional (default='human')
        The organism to retrieve data for. Choose 'human' or 'mouse'.

    copy : bool, optional (default=True)
        If True, return a copy of the AnnData object. If False, modify the object in place.

    drop_unmapped : bool, optional (default=False)
        If True, drop genes that could not be mapped to symbols.

    Returns
    -------
    AnnData
        The AnnData object with gene names transformed to gene symbols.
        If copy=True, this is a new object.

    Raises
    ------
    ValueError
        If not all genes in the AnnData object are in Ensembl ID format.
    """
    # Retrieve the Ensembl ID to gene symbol mapping
    ensembl2symbol = retrieve_ensembl2symbol_data(filename, organism)

    if not ensembl2symbol:
        raise ValueError("Failed to retrieve Ensembl ID to gene symbol mapping.")

    # Check if all genes are in Ensembl format
    all_ensembl = all(gene.startswith('ENS') for gene in adata.var_names)
    if not all_ensembl:
        raise ValueError("Not all genes are in Ensembl ID format. Please ensure all genes start with 'ENS'.")

    # Create a new AnnData object if copy is True, otherwise use the original
    adata_mod = adata.copy() if copy else adata

    # Create a mapping Series
    gene_map = pd.Series(ensembl2symbol)

    # Transform gene names
    new_var_names = adata_mod.var_names.map(gene_map)

    # Check if any genes were not mapped
    unmapped = new_var_names.isna()
    if unmapped.any():
        unmapped_count = unmapped.sum()
        print(f"Warning: {unmapped_count} genes could not be mapped to symbols.")

        if drop_unmapped:
            print(f"Dropping {unmapped_count} unmapped genes.")
            adata_mod = adata_mod[:, ~unmapped].copy()
            new_var_names = new_var_names[~unmapped]
        else:
            # For unmapped genes, keep the original Ensembl ID
            new_var_names = pd.Index([new_name if pd.notna(new_name) else old_name
                                      for new_name, old_name in zip(new_var_names, adata_mod.var_names)])

    # Assign new gene names
    adata_mod.var_names = new_var_names

    return adata_mod


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