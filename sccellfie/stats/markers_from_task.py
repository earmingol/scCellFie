import warnings
import pandas as pd


def get_task_determinant_genes(adata, metabolic_task, task_by_rxn, groupby=None, group=None, min_activity=0.0):
    """
    Finds the genes that determine the activity of all reactions in a metabolic task. Returns determinant genes
    for each reaction and their activity across specified cell groups, along with the fraction of cells in each
    group where the gene was determinant.

    Parameters
    ----------
    adata: AnnData object
        Annotated data matrix.

    metabolic_task: str
        Name of the metabolic task to analyze. Must be one of the tasks in the `task_by_rxn` DataFrame.
        It must also be present in the `adata.metabolic_tasks` attribute.

    task_by_rxn: pandas.DataFrame
        A pandas.DataFrame object where rows are metabolic tasks and columns are
        reactions. Each cell contains ones or zeros, indicating whether a reaction
        is involved in a metabolic task.

    groupby: str, optional (default: None)
        The key in the `adata.obs` DataFrame to group by. This could be any
        categorical annotation of cells (e.g., cell type, cluster).

    group: str or list, optional (default: None)
        The group(s) in the `adata.obs` DataFrame to analyze. If `None`, the analysis is performed
        by treating all single cells as a group. If `groups` is specified, `groupby` must be specified.
        The column referred by `groupby` must contain the groups specified in `group`.

    min_activity: float, optional (default: 0.0)
        Minimum reaction activity level to consider a reaction as active. Only genes that are
        associated with active reactions are considered. If zero, all reactions and therefore
        all genes are considered.

    Returns
    -------
    df: pandas.DataFrame
        A pandas.DataFrame reporting the determinant genes for each reaction in the metabolic task.
        The DataFrame has the following columns:
            - Group: The cell group.
            - Rxn: The reaction.
            - Det-Gene: The determinant gene for the reaction.
            - RAL: The reaction activity level for the reaction.
            - Cell_fraction: The fraction of cells in the group where this gene was determinant.

    Notes
    -----
    This function assumes that reaction activity levels have been computed using
    sccellfie.reaction_activity.compute_reaction_activity() and are stored in adata.reactions.X.

    Scores are computed as previously indicated in the CellFie paper (https://doi.org/10.1016/j.crmeth.2021.100040).
    """
    assert hasattr(adata, "metabolic_tasks"), "Please run scCellFie on your dataset before using this function."

    # Get list of rxns that belong to the metabolic task
    rxns_in_task = task_by_rxn.loc[metabolic_task, :]
    rxns_in_task = sorted([rxn for rxn in rxns_in_task[rxns_in_task != 0].index if rxn in adata.reactions.var_names])

    if (group is not None) & (groupby is None):
        warning_message = "You have specified `group` but not the column where to find the groups (`groupby`). Analysis will be performed across all groups."
        warnings.warn(warning_message, UserWarning)

    if groupby is not None:
        if group is not None:
            if isinstance(group, list):
                groups = group
            else:
                groups = [group]
        else:
            groups = adata.obs[groupby].unique().tolist()
        barcodes = [adata[adata.obs[groupby] == group].obs_names for group in groups]
    else:
        groups = ['All-Groups']
        barcodes = [adata.obs_names]

    dfs = []
    for _group, _barcodes in zip(groups, barcodes):
        # Get total number of cells in this group
        total_cells = len(_barcodes)

        rxn_filter = adata.reactions.obs_names.isin(_barcodes)
        adata_rxns = adata.reactions[rxn_filter]
        rxn_df = adata_rxns.to_df()
        adata_rxns.uns.update({'Rxn-Max-Genes': adata_rxns.uns['Rxn-Max-Genes'][rxn_filter]})

        for rxn in rxns_in_task:
            # Create initial DataFrame with cell-level information
            df = pd.DataFrame(index=_barcodes)
            df['Group'] = _group
            df['Rxn'] = rxn
            df['Det-Gene'] = adata_rxns.uns['Rxn-Max-Genes'][rxn]
            df['RAL'] = rxn_df[rxn]

            # Group by gene to get counts and mean RAL
            grouped = df.groupby(['Group', 'Rxn', 'Det-Gene']).agg({
                'RAL': 'mean',
                'Det-Gene': 'count'  # This gives us the count of cells for each gene
            }).rename(columns={'Det-Gene': 'Cell_count'})

            # Calculate fraction
            grouped['Cell_fraction'] = grouped['Cell_count'] / total_cells

            # Drop the cell count column as it was just for intermediate calculation
            grouped = grouped.drop('Cell_count', axis=1)

            dfs.append(grouped)

    df = pd.concat(dfs).reset_index().sort_values('RAL', ascending=False)
    if min_activity != 0.:
        df = df[df['RAL'] >= min_activity]
    df = df.reset_index(drop=True)
    return df