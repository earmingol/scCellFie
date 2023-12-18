import numpy as np
import scanpy as sc

def compute_mt_score(adata, task_by_rxn):
    assert hasattr(adata, 'reactions'), "You must run sccellfie.reaction_activity.compute_reaction_activity() first."

    # RAL matrix
    ral = adata.reactions.to_df()
    rxns = [rxn for rxn in task_by_rxn.columns if rxn in ral.columns]

    # Filter RAL matrix by reactions present in task_by_rxn
    ral = ral.loc[:, rxns]
    task_by_rxn = task_by_rxn.loc[:, rxns]

    # Multiply RAL matrix by transpose of task_by_rxn
    # This results in cells by sum of reaction activity per task (MTS matrix)
    mts = np.matmul(ral, task_by_rxn.T)

    # Keep only tasks that are present in all matrices
    tasks = [task for task in task_by_rxn.index if task in mts.columns]
    mts = mts.loc[:, tasks]
    task_by_rxn = task_by_rxn.loc[tasks, :]

    # Compute the average activity across reactions per task (MTS matrix)
    rxns_per_task = task_by_rxn.sum(axis=1)
    mts = mts.divide(rxns_per_task).fillna(0.0)

    # Remove tasks with zeros across cells
    mts = mts.loc[:, (mts != 0).any(axis=0)]
    drop_cols = [col for col in mts.columns if col in adata.obs.columns]
    adata.metabolic_tasks = sc.AnnData(mts,
                                       obs=adata.obs.drop(columns=drop_cols),
                                       obsm=adata.obsm,
                                       obsp=adata.obsp,
                                       uns=adata.uns
                                       )

