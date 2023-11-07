import numpy as np


def compute_mt_score(adata, task_by_rxn):
    assert 'RAL' in adata.uns.keys(), "You must run sccellfie.reaction_activity.compute_reaction_activity() first."

    ral = adata.uns['RAL']
    rxns = [rxn for rxn in task_by_rxn.columns if rxn in ral.columns]

    ral = ral.loc[:, rxns]
    task_by_rxn = task_by_rxn.loc[:, rxns]

    mts = np.matmul(task_by_rxn, ral.T).T

    tasks = [task for task in task_by_rxn.index if task in mts.columns]
    mts = mts.loc[:, tasks]
    task_by_rxn = task_by_rxn.loc[tasks, :]
    rxns_per_task = task_by_rxn.sum(axis=1)
    mts = mts.divide(rxns_per_task).fillna(0.0)
    # Remove tasks with zeros across cells
    mts = mts.loc[:, (mts != 0).any(axis=0)]
    adata.uns.update({'MT_scores' : mts})


