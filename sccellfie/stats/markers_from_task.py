import warnings
import pandas as pd


def get_task_determinant_genes(adata, metabolic_task, task_by_rxn, groupby=None, group=None, min_activity=0.0):
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
        rxn_filter = adata.reactions.obs_names.isin(_barcodes)
        adata_rxns = adata.reactions[rxn_filter]
        rxn_df = adata_rxns.to_df()
        adata_rxns.uns.update({'Rxn-Max-Genes': adata_rxns.uns['Rxn-Max-Genes'][rxn_filter]})

        for rxn in rxns_in_task:
            df = pd.DataFrame(index=_barcodes)
            df['Group'] = _group
            df['Rxn'] = rxn
            df['Det-Gene'] = adata_rxns.uns['Rxn-Max-Genes'][rxn]
            df['RAL'] = rxn_df[rxn]
            df = df.groupby(['Group', 'Rxn', 'Det-Gene']).mean()
            dfs.append(df)
    df = pd.concat(dfs).reset_index().sort_values('RAL', ascending=False)
    if min_activity != 0.:
        df = df[df['RAL'] > min_activity]
    return df