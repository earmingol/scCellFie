import pytest
import pandas as pd
from unittest.mock import patch
from io import StringIO

from sccellfie.datasets.gene_info import retrieve_ensembl2symbol_data

# Mock data for testing
MOCK_HUMAN_DATA = """ensembl_id,symbol
ENSG00000139618,BRCA2
ENSG00000141510,TP53
"""

MOCK_MOUSE_DATA = """ensembl_id,symbol
ENSMUSG00000041147,Brca2
ENSMUSG00000059552,Trp53
"""

def test_default_human():
    with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(MOCK_HUMAN_DATA))):
        result = retrieve_ensembl2symbol_data()
    assert result == {'ENSG00000139618': 'BRCA2', 'ENSG00000141510': 'TP53'}

def test_default_mouse():
    with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(MOCK_MOUSE_DATA))):
        result = retrieve_ensembl2symbol_data(organism='mouse')
    assert result == {'ENSMUSG00000041147': 'Brca2', 'ENSMUSG00000059552': 'Trp53'}

def test_custom_file():
    mock_file_content = MOCK_HUMAN_DATA
    with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(mock_file_content))):
        result = retrieve_ensembl2symbol_data(filename='custom_file.csv')
    assert result == {'ENSG00000139618': 'BRCA2', 'ENSG00000141510': 'TP53'}

def test_invalid_organism():
    with pytest.raises(ValueError, match="Invalid organism"):
        retrieve_ensembl2symbol_data(organism='invalid')

def test_file_not_found():
    with patch('pandas.read_csv', side_effect=FileNotFoundError):
        result = retrieve_ensembl2symbol_data()
    assert result == {}

def test_empty_file():
    with patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError):
        result = retrieve_ensembl2symbol_data()
    assert result == {}

def test_missing_columns():
    invalid_data = """col1,col2
    data1,data2
    """
    with patch('pandas.read_csv', return_value=pd.read_csv(StringIO(invalid_data))):
        result = retrieve_ensembl2symbol_data()
    assert result == {}  # Expecting an empty dictionary instead of raising an error

def test_general_exception():
    with patch('pandas.read_csv', side_effect=Exception("Test exception")):
        result = retrieve_ensembl2symbol_data()
    assert result == {}