import numpy as np
import scanpy as sc


def compute_mt_score(adata, task_by_rxn, verbose=True):
    """
    Computes the metabolic task score for each cell in an AnnData object given
    specific reaction activity levels.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    task_by_rxn: pandas.DataFrame
        A pandas.DataFrame object where rows are metabolic tasks and columns are
        reactions. Each cell contains ones or zeros, indicating whether a reaction
        is involved in a metabolic task.

    verbose: bool, optional (default: True)
        Whether to print information about the analysis.

    Returns
    -------
    None
        An AnnData object is added to the adata object in adata.metabolic_tasks. This object
        contains the metabolic task scores for each cell in adata.obs_names. Here,
        each metabolic task is an element of var_name in adata.metabolic_tasks. The metabolic task scores
        are stored in adata.metabolic_tasks.X.

    Notes
    -----
    This function assumes that reaction activity levels have been computed using
    sccellfie.reaction_activity.compute_reaction_activity() and are stored in adata.reactions.X.

    This score is computed as previously indicated in the CellFie paper (https://doi.org/10.1016/j.crmeth.2021.100040).
    """
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
    old_cols = set(mts.columns)
    mts = mts.loc[:, (mts != 0).any(axis=0)]
    new_cols = set(mts.columns)
    removed_cols = old_cols.difference(new_cols)
    if verbose:
        print(f"Removed {len(removed_cols)} metabolic tasks with zeros across all cells.")

    # Prepare output
    drop_cols = [col for col in mts.columns if col in adata.obs.columns]
    adata.metabolic_tasks = sc.AnnData(mts,
                                       obs=adata.obs.drop(columns=drop_cols).copy(),
                                       obsm=adata.obsm.copy(),
                                       obsp=adata.obsp.copy(),
                                       uns=adata.uns.copy()
                                       )

