import pytest
import os
import tempfile
import pandas as pd

from unittest.mock import patch

from sccellfie.datasets.database import load_sccellfie_database  # Replace 'your_module' with the actual module name

# Mock data for testing
mock_json_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
mock_csv_data = pd.DataFrame({'col1': [4, 5, 6], 'col2': ['d', 'e', 'f']})


@pytest.fixture
def mock_read_json(monkeypatch):
    def mock_read(path):
        return mock_json_data

    monkeypatch.setattr(pd, 'read_json', mock_read)


@pytest.fixture
def mock_read_csv(monkeypatch):
    def mock_read(path, index_col=None):
        return mock_csv_data

    monkeypatch.setattr(pd, 'read_csv', mock_read)


def test_load_sccellfie_database_default_urls(mock_read_json, mock_read_csv):
    data = load_sccellfie_database(organism='human')
    assert isinstance(data, dict)
    assert 'rxn_info' in data
    assert 'task_info' in data
    assert 'task_by_rxn' in data
    assert 'task_by_gene' in data
    assert 'rxn_by_gene' in data
    assert 'thresholds' in data
    assert data['organism'] == 'human'
    assert data['rxn_info'].equals(mock_json_data)
    assert data['task_info'].equals(mock_csv_data)


def test_load_sccellfie_database_local_folder():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create mock files
        pd.DataFrame().to_json(os.path.join(tmpdirname, 'Rxn-Info-Recon2-2.json'))
        pd.DataFrame().to_csv(os.path.join(tmpdirname, 'Task-Info.csv'))
        pd.DataFrame().to_csv(os.path.join(tmpdirname, 'Task_by_Rxn.csv'))
        pd.DataFrame().to_csv(os.path.join(tmpdirname, 'Task_by_Gene.csv'))
        pd.DataFrame().to_csv(os.path.join(tmpdirname, 'Rxn_by_Gene.csv'))
        pd.DataFrame().to_csv(os.path.join(tmpdirname, 'Thresholds.csv'))

        data = load_sccellfie_database(organism='human', task_folder=tmpdirname)
        assert isinstance(data, dict)
        assert 'rxn_info' in data
        assert 'task_info' in data
        assert 'task_by_rxn' in data
        assert 'task_by_gene' in data
        assert 'rxn_by_gene' in data
        assert 'thresholds' in data
        assert data['organism'] == 'human'


def test_load_sccellfie_database_individual_files():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create mock files with unique names
        rxn_info_path = os.path.join(tmpdirname, 'custom_rxn_info.json')
        task_info_path = os.path.join(tmpdirname, 'custom_task_info.csv')
        task_by_rxn_path = os.path.join(tmpdirname, 'custom_task_by_rxn.csv')
        task_by_gene_path = os.path.join(tmpdirname, 'custom_task_by_gene.csv')
        rxn_by_gene_path = os.path.join(tmpdirname, 'custom_rxn_by_gene.csv')
        thresholds_path = os.path.join(tmpdirname, 'custom_thresholds.csv')

        pd.DataFrame().to_json(rxn_info_path)
        pd.DataFrame().to_csv(task_info_path)
        pd.DataFrame().to_csv(task_by_rxn_path)
        pd.DataFrame().to_csv(task_by_gene_path)
        pd.DataFrame().to_csv(rxn_by_gene_path)
        pd.DataFrame().to_csv(thresholds_path)

        data = load_sccellfie_database(
            organism='human',
            rxn_info_filename=rxn_info_path,
            task_info_filename=task_info_path,
            task_by_rxn_filename=task_by_rxn_path,
            task_by_gene_filename=task_by_gene_path,
            rxn_by_gene_filename=rxn_by_gene_path,
            thresholds_filename=thresholds_path
        )
        assert isinstance(data, dict)
        assert 'rxn_info' in data
        assert 'task_info' in data
        assert 'task_by_rxn' in data
        assert 'task_by_gene' in data
        assert 'rxn_by_gene' in data
        assert 'thresholds' in data
        assert data['organism'] == 'human'


def test_load_sccellfie_database_invalid_organism():
    with pytest.raises(ValueError):
        load_sccellfie_database(organism='invalid')


@patch('pandas.read_json')
@patch('pandas.read_csv')
def test_load_sccellfie_database_file_error(mock_read_csv, mock_read_json):
    mock_read_json.side_effect = Exception("Mock JSON read error")
    mock_read_csv.side_effect = Exception("Mock CSV read error")

    data = load_sccellfie_database(organism='human')
    assert isinstance(data, dict)
    assert all(value is None for key, value in data.items() if key != 'organism')
    assert data['organism'] == 'human'