import scanpy as sc
import numpy as np
import pandas as pd

from collections import defaultdict
from tqdm import tqdm

from sccellfie.gene_score import compute_gpr_gene_score


def compute_reaction_activity(adata, gpr_dict, use_specificity=True, layer='gene_scores', disable_pbar=False):
    """
    Computes reaction activity from gene scores and GPRs.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    gpr_dict: dict
        A dictionary with reaction IDs as keys and Gene-Protein Rules (GPRs) as values.
        GPRs are in the format of the ast library after initialization with cobra from strings
        to tree format for AND and OR rules.

    use_specificity: bool, optional (default: True)
        Whether to use the specificity of the determinant gene to compute reaction
        activity levels. The specificity is the inverse of the number of reactions
        that the determinant gene is involved in.

    layer: str, optional (default: 'gene_scores')
        The name of the layer in adata where the gene scores are stored.

    disable_pbar: bool, optional (default: False)
        Whether to disable the progress bar.

    Returns
    -------
    None
        An AnnData object is added to the adata object in adata.reactions. This object
        contains the reaction activity levels for each cell in adata.obs_names. Here,
        each reaction is an element of var_name in adata.reactions. The reaction scores
        are stored in adata.reactions.X, and the determinant gene for each reaction is
        stored in adata.reactions.uns['Rxn-Max-Genes'].

    Notes
    -----
    This function assumes that gene scores have been computed using
    sccellfie.gene_score.compute_gene_scores() and are stored in adata.layers['gene_scores'].

    This function assumes that the GPRs are valid in COBRApy. Also, that the GPRs are in the
    form of a string, e.g., '(gene1 and gene2) or gene3'.

    This score is computed as previously indicated in the CellFie paper (https://doi.org/10.1016/j.crmeth.2021.100040).
    """
    genes = adata.var_names
    gene_scores = adata.layers[layer]
    rxns = gpr_dict.keys()
    ral = np.zeros((gene_scores.shape[0], len(rxns)))

    # This could be optimized by paralellization, returning multiple vectors (one per cell)
    # And concatenating them later.
    rxn_max_genes = []
    for i in tqdm(range(gene_scores.shape[0]), disable=disable_pbar, desc='Cell Rxn Activities', leave=True): # Iterate through single cells
        max_gene_vector = []
        scores = defaultdict(float)
        scores.update({name: value for name, value in zip(genes, gene_scores[i, :])})

        # For accounting for specificity
        selected_gene = defaultdict(float)
        rxn_ids_gene = defaultdict(list)

        for j, k in enumerate(rxns):
            gpr = gpr_dict[k]
            score, gene = compute_gpr_gene_score(gpr, scores)
            ral[i, j] = score
            selected_gene[gene] += 1
            rxn_ids_gene[gene].append(j)
            max_gene_vector.append(gene)

        rxn_max_genes.append(max_gene_vector)
        # Multiply RAL by S (specificity of determinant gene)
        if use_specificity:
            for g, times in selected_gene.items():
                if times == 0.0:
                    times = 1.0
                ral[i, rxn_ids_gene[g]] /= times


    ral_df = pd.DataFrame(ral, index=adata.obs_names, columns=rxns)
    drop_cols = [col for col in ral_df.columns if col in adata.obs.columns]
    adata.reactions = sc.AnnData(ral_df,
                                 obs=adata.obs.drop(columns=drop_cols).copy(),
                                 obsm=adata.obsm.copy(),
                                 obsp=adata.obsp.copy(),
                                 uns=adata.uns.copy()
                                 )

    rxn_max_genes = np.asarray(rxn_max_genes)
    rxn_max_genes = pd.DataFrame(rxn_max_genes, index=adata.obs_names, columns=rxns)
    adata.reactions.uns.update({'Rxn-Max-Genes' : rxn_max_genes.astype(str)})


# THIS IS FOR PARALLEL PROCESSING - IT WORKS SLOWER THAN SINGLE CORE FUNCTIONS. TODO: TRY TO MAKE IT FASTER
#
# import concurrent
# from concurrent.futures import ProcessPoolExecutor
# from typing import Union, Optional, Tuple, Collection, Sequence, Iterable, Literal
#
# def reaction_activity_worker(i, gene_scores, genes, gpr_dict):
#     scores = defaultdict(float)
#     scores.update({name: value for name, value in zip(genes, gene_scores[i, :])})
#     ral = np.zeros(len(gpr_dict))  # Array to hold reaction activity levels for cell 'i'
#     selected_gene = defaultdict(float)
#     rxn_ids_gene = defaultdict(list)
#     rxn_max_genes = []  # List to hold the maximum scored gene for each reaction
#
#     # Compute reaction activity levels for each reaction
#     for j, (rxn_name, gpr) in enumerate(gpr_dict.items()):
#         try:  # Fix case when GPR is not valid in COBRA
#             gpr_parsed = cobra.core.gene.GPR().from_string(gpr)
#             score, gene = evaluate_gene_score(gpr_parsed, scores)
#         except Exception as e:
#             continue
#         ral[j] = score
#         selected_gene[gene] += 1
#         rxn_ids_gene[gene].append(j)
#         rxn_max_genes.append((rxn_name, gene))
#
#     return i, ral, selected_gene, rxn_ids_gene, rxn_max_genes
#
# def compute_reaction_activity(adata, gpr_dict, use_specificity=True, disable_pbar=False):
#     genes = adata.var_names
#     gene_scores = adata.layers['gene_scores']
#     rxns = list(gpr_dict.keys())
#
#     # Prepare the data structure for parallel processing
#     futures = []
#     ral_list = np.zeros((gene_scores.shape[0], len(rxns)))  # Initialize the RAL array
#
#     # Use parallel processing to compute reaction activities
#     with ProcessPoolExecutor() as executor:
#         for i in range(gene_scores.shape[0]):  # Iterate through single cells
#             future = executor.submit(reaction_activity_worker, i, gene_scores, genes, gpr_dict)
#             futures.append(future)
#
#         # Collect the results as they come in
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), disable=disable_pbar):
#             i, ral, selected_gene, rxn_ids_gene, rxn_max_genes = future.result()
#             ral_list[i, :] = ral
#             # Use specificity if required
#             if use_specificity:
#                 for gene, times in selected_gene.items():
#                     if times == 0.0: times = 1.0  # Avoid division by zero
#                     ral_list[i, rxn_ids_gene[gene]] /= times
#
#     # Convert RAL and max genes to DataFrame and update 'adata.uns'
#     ral_df = pd.DataFrame(ral_list, index=adata.obs_names, columns=rxns)
#     rxn_max_genes_df = pd.DataFrame([item for future in futures for item in future.result()[4]], columns=['Rxn', 'Max-Scored-Gene'])
#     adata.uns['RAL'] = ral_df
#     adata.uns['Rxn-Max-Genes'] = rxn_max_genes_df
