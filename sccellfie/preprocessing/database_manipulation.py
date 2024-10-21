import numpy as np
import pandas as pd
from sccellfie.preprocessing.gpr_rules import find_genes_gpr


def get_element_associations(df, element, axis_element=0):
    """
    Gets the tasks, reactions, or genes associated with
    a given element in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the associations.

    element : str
        Element for which to get the associations. This can be a task, reaction, or gene.
        Name should match exactly the name in indexes or columns of the DataFrame.

    axis_element : int, optional (default: 0)
        Axis along which the element is located. Can be 0 (rows) or 1 (columns).

    Returns
    -------
    associations : list of str
        List of tasks, reactions, or genes associated with the given element.

    """
    if axis_element == 0:
        e = df.loc[element, :]
    elif axis_element == 1:
        e = df.loc[:, element]
    else:
        raise ValueError('Not a valid axis')

    e = e.loc[e != 0]
    associations = sorted(e.index)
    return associations


def add_new_task(task_by_rxn, task_by_gene, rxn_by_gene, task_info, rxn_info,
                 task_name, task_system, task_subsystem, rxn_names, gpr_hgncs, gpr_symbols):
    """
    Adds a new task and their associated reactions and genes to the database.

    Parameters
    ----------
    task_by_rxn : pandas.DataFrame
        DataFrame representing the relationship between tasks and reactions.

    task_by_gene : pandas.DataFrame
        DataFrame representing the relationship between tasks and genes.

    rxn_by_gene : pandas.DataFrame
        DataFrame representing the relationship between reactions and genes.

    task_info : pandas.DataFrame
        DataFrame containing information about tasks, including the task name,
        system (major group of tasks), and subsystem (specific group of tasks).

    rxn_info : pandas.DataFrame
        DataFrame containing information about reactions, including the reaction name,
        and the associated GPR rules in HGNC and symbol format.

    task_name : str
        Name of the task to add.

    task_system : str
        System (major group of tasks) to which the task belongs.

    task_subsystem : str
        Subsystem (specific group of tasks) to which the task belongs.

    rxn_names : list of str
        List of reaction names associated with the task.

    gpr_hgncs : list of str
        List of GPR rules in HGNC format associated with the reactions. Order
        should match the order of the reaction names.

    gpr_symbols : list of str
        List of GPR rules in symbol format associated with the reactions. Order
        should match the order of the reaction names.

    Returns
    -------
    task_by_rxn : pandas.DataFrame
        Updated DataFrame representing the relationship between tasks and reactions.

    task_by_gene : pandas.DataFrame
        Updated DataFrame representing the relationship between tasks and genes.

    rxn_by_gene : pandas.DataFrame
        Updated DataFrame representing the relationship between reactions and genes.

    task_info : pandas.DataFrame
        Updated DataFrame containing information about tasks, including the task name,
        system (major group of tasks), and subsystem (specific group of tasks).

    rxn_info : pandas.DataFrame
        Updated DataFrame containing information about reactions, including the reaction name,
        and the associated GPR rules in HGNC and symbol format.
    """
    task_by_rxn, task_by_gene, rxn_by_gene, task_info, rxn_info = task_by_rxn.copy(), task_by_gene.copy(), rxn_by_gene.copy(), task_info.copy(), rxn_info.copy(),
    # Add task
    if (task_name not in task_by_rxn.index):
        task_by_rxn.loc[task_name] = task_by_rxn.shape[1] * [0]
    if (task_name not in task_by_gene.index):
        task_by_gene.loc[task_name] = task_by_gene.shape[1] * [0]

    # Annotate task
    if task_name not in task_info.index:
        task_info.loc[len(task_info)] = [task_name, task_system, task_subsystem]

    # Add rxns
    for rxn_name, gpr_hgnc, gpr_symbol in zip(rxn_names, gpr_hgncs, gpr_symbols):
        # Add to GPR rules
        if rxn_name not in rxn_info.Reaction.values.tolist():
            rxn_info.loc[len(rxn_info)] = [rxn_name, gpr_hgnc, gpr_symbol]

        # Add rxn to task_by_rxn
        df = task_by_rxn
        if rxn_name not in df.columns:
            df[rxn_name] = [0] * df.shape[0]
        df.loc[task_name, rxn_name] = 1

        # Add rxn and gene to rxn_by_gene
        df = rxn_by_gene
        if rxn_name not in df.index:
            df.loc[rxn_name] = [0] * df.shape[1]

        for gene in find_genes_gpr(gpr_symbol):
            if gene not in df.columns:
                df[gene] = [0] * df.shape[0]
            df.loc[rxn_name, gene] = 1

        # Add gene to task_by_gene
        df = task_by_gene
        for gene in find_genes_gpr(gpr_symbol):
            if gene not in df.columns:
                df[gene] = [0] * df.shape[0]
            df.loc[task_name, gene] = 1

    return task_by_rxn, task_by_gene, rxn_by_gene, task_info, rxn_info


def combine_and_sort_dataframes(df1, df2, preference='max'):
    """
    Combines two DataFrames and sort the rows and columns alphabetically.

    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame to combine.

    df2 : pandas.DataFrame
        Second DataFrame to combine.

    preference : str, optional
        Preference for which value to keep when both dataframes have the same cell.
        Options: 'max' (default), 'min', 'df1', 'df2'.

    Returns
    -------
    combined_df : pandas.DataFrame
        Combined DataFrame with all rows and columns from df1 and df2, sorted alphabetically.
        Missing values are filled with 0.
    """
    # Get the union of index (rows) and columns
    all_rows = df1.index.union(df2.index)
    all_columns = df1.columns.union(df2.columns)

    # Create a new DataFrame with all rows and columns, filled with NaN
    combined_df = pd.DataFrame(np.nan, index=all_rows, columns=all_columns)

    # Update the combined DataFrame with values from df1
    combined_df.update(df1)

    # Reindex df2 to match the combined DataFrame's structure
    df2_reindexed = df2.reindex(index=all_rows, columns=all_columns)

    if preference == 'max':
        combined_df = combined_df.combine(df2_reindexed, np.fmax)
    elif preference == 'min':
        combined_df = combined_df.combine(df2_reindexed, np.fmin)
    elif preference == 'df1':
        combined_df.update(df2_reindexed, overwrite=False)
    elif preference == 'df2':
        combined_df.update(df2_reindexed)
    else:
        raise ValueError("Invalid preference. Choose 'max', 'min', 'df1', or 'df2'.")

    # Sort the rows and columns alphabetically
    combined_df = combined_df.sort_index().sort_index(axis=1).fillna(0)

    return combined_df


def handle_duplicate_indexes(df, value_column=None, operation='first'):
    """
    Handles duplicated indexes in a DataFrame by keeping the min, max, mean, first, or last value
    associated with them in a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with duplicated indexes.

    value_column : str, optional (default: None)
        Name of the column containing values to make a decision
         when handling duplicated indexes. This value is optional
         only when operation is 'first' or 'last'.

    operation : str, optional (default: 'first')
        Operation to perform when handling duplicated indexes.
        Options: 'min', 'max', 'mean', 'first', 'last'.

    Returns
    -------
    df_result : pandas.DataFrame
        DataFrame with duplicated indexes handled according to the specified operation
    """
    if df.empty:
        return df.copy()

    if operation not in ['min', 'max', 'mean', 'first', 'last']:
        raise ValueError("Operation must be 'min', 'max', 'mean', or 'first'")

    if operation in ['first', 'last']:
        return df[~df.index.duplicated(keep=operation)]

    # Group by index and apply the specified operation
    assert value_column is not None, "A value column must be provided for operations other than 'first' or 'last'"
    if operation == 'mean':
        df_grouped = df.groupby(level=0).agg({value_column: 'mean'})
    else:  # min or max
        df_grouped = df.groupby(level=0).agg({value_column: operation})

    # Merge the result back with the original DataFrame to keep other columns
    df_result = df.loc[~df.index.duplicated(keep='first')].copy()
    df_result[value_column] = df_grouped[value_column]
    return df_result