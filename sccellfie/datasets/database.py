import pandas as pd
import os


def load_sccellfie_database(organism='human', task_folder=None, rxn_info_filename=None, task_info_filename=None,
                            task_by_rxn_filename=None, task_by_gene_filename=None, rxn_by_gene_filename=None,
                            thresholds_filename=None):
    """
    Loads files of the metabolic task database from either a local folder, individual file paths, or predefined URLs.

    Parameters
    ----------
    organism : str, optional (default: 'human')
        The organism to retrieve data for. Choose 'human' or 'mouse'. Used when loading from URLs.

    task_folder : str, optional (default: None)
        The local folder path containing CellFie data files. If provided, this takes priority.

    rxn_info_filename : str, optional (default: None)
        Full path for reaction information JSON file.

    task_info_filename : str, optional (default: None)
        Full path for task information CSV file.

    task_by_rxn_filename : str, optional (default: None)
        Full path for task by reaction CSV file.

    task_by_gene_filename : str, optional (default: None)
        Full path for task by gene CSV file.

    rxn_by_gene_filename : str, optional (default: None)
        Full path for reaction by gene CSV file.

    thresholds_filename : str, optional (default: None)
        Full path for thresholds CSV file.

    Returns
    -------
    data : dict
        A dictionary containing the loaded data frames and information.
        Keys are 'rxn_info', 'task_info', 'task_by_rxn', 'task_by_gene', 'rxn_by_gene',
        'thresholds', and 'organism'.
        Examples of dataframes can be found at https://github.com/earmingol/scCellFie/raw/refs/heads/main/task_data/homo_sapiens/
    """
    # Define default URLs for human and mouse data
    default_urls = {
        'human': 'https://github.com/earmingol/scCellFie/raw/refs/heads/main/task_data/homo_sapiens/',
        'mouse': 'https://github.com/earmingol/scCellFie/raw/refs/heads/main/task_data/mus_musculus/'
    }

    # Define default file names
    default_file_names = {
        'human': {
            'rxn_info': 'Rxn-Info-Recon2-2.json',
            'task_info': 'Task-Info.csv',
            'task_by_rxn': 'Task_by_Rxn.csv',
            'task_by_gene': 'Task_by_Gene.csv',
            'rxn_by_gene': 'Rxn_by_Gene.csv',
            'thresholds': 'Thresholds.csv'
        },
        'mouse': {
            'rxn_info': 'Rxn-Info-iMM1415.json',
            'task_info': 'Task-Info.csv',
            'task_by_rxn': 'Task_by_Rxn.csv',
            'task_by_gene': 'Task_by_Gene.csv',
            'rxn_by_gene': 'Rxn_by_Gene.csv',
            'thresholds': 'Thresholds.csv'
        }
    }

    # Determine the base path and file names
    if task_folder:
        base_path = task_folder
        file_paths = {
            'rxn_info': os.path.join(base_path, default_file_names[organism]['rxn_info']),
            'task_info': os.path.join(base_path, default_file_names[organism]['task_info']),
            'task_by_rxn': os.path.join(base_path, default_file_names[organism]['task_by_rxn']),
            'task_by_gene': os.path.join(base_path, default_file_names[organism]['task_by_gene']),
            'rxn_by_gene': os.path.join(base_path, default_file_names[organism]['rxn_by_gene']),
            'thresholds': os.path.join(base_path, default_file_names[organism]['thresholds'])
        }
    else:
        base_path = default_urls.get(organism.lower())
        if not base_path:
            raise ValueError("Invalid organism. Choose 'human' or 'mouse', or provide a custom folder path.")
        file_paths = {
            'rxn_info': rxn_info_filename or f"{base_path}/{default_file_names[organism]['rxn_info']}",
            'task_info': task_info_filename or f"{base_path}/{default_file_names[organism]['task_info']}",
            'task_by_rxn': task_by_rxn_filename or f"{base_path}/{default_file_names[organism]['task_by_rxn']}",
            'task_by_gene': task_by_gene_filename or f"{base_path}/{default_file_names[organism]['task_by_gene']}",
            'rxn_by_gene': rxn_by_gene_filename or f"{base_path}/{default_file_names[organism]['rxn_by_gene']}",
            'thresholds': thresholds_filename or f"{base_path}/{default_file_names[organism]['thresholds']}"
        }

    # Function to load a file
    def load_file(file_key, index_col=None):
        full_path = file_paths[file_key]
        try:
            if full_path.endswith('.json'):
                return pd.read_json(full_path)
            elif full_path.endswith('.csv'):
                return pd.read_csv(full_path, index_col=index_col)
            else:
                raise ValueError(f"Unsupported file format: {full_path}")
        except Exception as e:
            print(f"Error loading {full_path}: {str(e)}")
            return None

    # Load all files
    data = {}
    data['rxn_info'] = load_file('rxn_info')
    data['task_info'] = load_file('task_info')
    data['task_by_rxn'] = load_file('task_by_rxn', index_col='Task')
    data['task_by_gene'] = load_file('task_by_gene', index_col='Task')
    data['rxn_by_gene'] = load_file('rxn_by_gene', index_col='Reaction')
    data['thresholds'] = load_file('thresholds', index_col='symbol')
    data['organism'] = organism
    return data