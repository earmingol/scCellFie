import cobra
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from .gene_score import evaluate_gene_score
# from typing import Union, Optional, Tuple, Collection, Sequence, Iterable, Literal


def compute_reaction_activity(adata, gpr_dict, use_specificity=True, disable_pbar=False):
    '''Computing reaction activity from gene scores and GPRs'''
    genes = adata.var_names
    gene_scores = adata.layers['gene_scores']
    rxns = gpr_dict.keys()
    ral = np.zeros((gene_scores.shape[0], len(rxns)))
    # This could be optimized by paralellization, returning multiple vectors (one per cell)
    # And concatenating them later.
    for i in tqdm(range(gene_scores.shape[0]), disable=disable_pbar): # Iterate through single cells
        scores = defaultdict(float)
        scores.update({name: value for name, value in zip(genes, gene_scores[i, :])})

        # For accounting for specificity
        selected_gene = defaultdict(float)
        rxn_ids_gene = defaultdict(list)

        for j, k in enumerate(rxns):
            gpr = gpr_dict[k]
            try: # Fix case when GPR is not valid in COBRA
                score, gene = evaluate_gene_score(cobra.core.gene.GPR().from_string(gpr), scores)
            except:
                continue
            ral[i, j] = score
            selected_gene[gene] += 1
            rxn_ids_gene[gene].append(j)

        # Multiply RAL by S (specificity of determinant gene)
        if use_specificity:
            for g, times in selected_gene.items():
                if times == 0.0:
                    times = 1.0
                ral[i, rxn_ids_gene[g]] /= times

    ral_df = pd.DataFrame(ral, index=adata.obs_names, columns=rxns)
    adata.uns.update({'RAL' : ral_df})


