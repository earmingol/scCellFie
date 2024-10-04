import pytest
import pandas as pd

from sccellfie.preprocessing.database_manipulation import get_element_associations, add_new_task, combine_and_sort_dataframes


def test_get_element_associations():
    # Create a sample DataFrame
    df = pd.DataFrame({
        'R1': [1, 0, 1],
        'R2': [1, 1, 0],
        'R3': [0, 1, 1]
    }, index=['T1', 'T2', 'T3'])

    # Test getting associations along rows (axis=0)
    assert get_element_associations(df, 'T1', axis_element=0) == ['R1', 'R2']
    assert get_element_associations(df, 'T2', axis_element=0) == ['R2', 'R3']

    # Test getting associations along columns (axis=1)
    assert get_element_associations(df, 'R1', axis_element=1) == ['T1', 'T3']
    assert get_element_associations(df, 'R3', axis_element=1) == ['T2', 'T3']

    # Test with invalid axis
    with pytest.raises(ValueError):
        get_element_associations(df, 'T1', axis_element=2)


def test_add_new_task():
    # Create sample DataFrames
    task_by_rxn = pd.DataFrame({'R1': [1, 0], 'R2': [1, 1]}, index=['T1', 'T2'])
    task_by_gene = pd.DataFrame({'G1': [1, 0], 'G2': [1, 1]}, index=['T1', 'T2'])
    rxn_by_gene = pd.DataFrame({'G1': [1, 0], 'G2': [1, 1]}, index=['R1', 'R2'])
    task_info = pd.DataFrame({'Task': ['T1', 'T2'], 'System': ['S1', 'S1'], 'Subsystem': ['SS1', 'SS2']})
    rxn_info = pd.DataFrame({'Reaction': ['R1', 'R2'], 'GPR_HGNC': ['HGNC:1', 'HGNC:2 and HGNC:3'], 'GPR_Symbol': ['G1', 'G2 and G3']})

    # Add a new task
    new_task_by_rxn, new_task_by_gene, new_rxn_by_gene, new_task_info, new_rxn_info = add_new_task(
        task_by_rxn, task_by_gene, rxn_by_gene, task_info, rxn_info,
        'T3', 'S2', 'SS3', ['R3'], ['HGNC:4 or HGNC:5'], ['G4 or G5']
    )

    # Check if the new task is added correctly
    assert 'T3' in new_task_by_rxn.index
    assert 'T3' in new_task_by_gene.index
    assert 'R3' in new_task_by_rxn.columns
    assert 'G4' in new_task_by_gene.columns and 'G5' in new_task_by_gene.columns
    assert 'R3' in new_rxn_by_gene.index
    assert 'G4' in new_rxn_by_gene.columns and 'G5' in new_rxn_by_gene.columns
    assert 'T3' in new_task_info['Task'].values
    assert 'R3' in new_rxn_info['Reaction'].values


def test_combine_and_sort_dataframes():
    # Create sample DataFrames
    df1 = pd.DataFrame({'B': [1, 2], 'A': [3, 4]}, index=['Y', 'X'])
    df2 = pd.DataFrame({'C': [5, 6], 'A': [7, 8]}, index=['Z', 'X'])

    # Test with default preference (max)
    result_max = combine_and_sort_dataframes(df1, df2)
    expected_result_max = pd.DataFrame({
        'A': [8, 3, 7],
        'B': [2, 1, 0],
        'C': [6, 0, 5]
    }, index=['X', 'Y', 'Z'], dtype=float)
    pd.testing.assert_frame_equal(result_max, expected_result_max)

    # Test with 'min' preference
    result_min = combine_and_sort_dataframes(df1, df2, preference='min')
    expected_result_min = pd.DataFrame({
        'A': [4, 3, 7],
        'B': [2, 1, 0],
        'C': [6, 0, 5]
    }, index=['X', 'Y', 'Z'], dtype=float)
    pd.testing.assert_frame_equal(result_min, expected_result_min)

    # Test with 'df1' preference
    result_df1 = combine_and_sort_dataframes(df1, df2, preference='df1')
    expected_result_df1 = pd.DataFrame({
        'A': [4, 3, 7],
        'B': [2, 1, 0],
        'C': [6, 0, 5]
    }, index=['X', 'Y', 'Z'], dtype=float)
    pd.testing.assert_frame_equal(result_df1, expected_result_df1)

    # Test with 'df2' preference
    result_df2 = combine_and_sort_dataframes(df1, df2, preference='df2')
    expected_result_df2 = pd.DataFrame({
        'A': [8, 3, 7],
        'B': [2, 1, 0],
        'C': [6, 0, 5]
    }, index=['X', 'Y', 'Z'], dtype=float)
    pd.testing.assert_frame_equal(result_df2, expected_result_df2)

    # Test with empty DataFrames
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    result_empty = combine_and_sort_dataframes(df3, df4)
    assert result_empty.empty

    # Test with invalid preference
    with pytest.raises(ValueError):
        combine_and_sort_dataframes(df1, df2, preference='invalid')