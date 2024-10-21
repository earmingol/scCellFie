import pandas as pd


def retrieve_ensembl2symbol_data(filename=None, organism='human'):
    """
    Retrieves a dictionary mapping Ensembl IDs to gene symbols for a given organism.

    Parameters
    ----------
    filename : str, optional (default: None)
        The file path to a custom CSV file containing Ensembl IDs and gene symbols.

    organism : str, optional (default: 'human')
        The organism to retrieve data for. Choose 'human' or 'mouse'.

    Returns
    -------
    ensembl2symbol : dict
        A dictionary mapping Ensembl IDs to gene symbols
    """
    # Define default URLs for human and mouse data
    default_urls = {
        'human': 'https://github.com/earmingol/scCellFie/raw/refs/heads/main/task_data/Ensembl2symbol_human.csv',
        'mouse': 'https://github.com/earmingol/scCellFie/raw/refs/heads/main/task_data/Ensembl2symbol_mouse.csv'
    }

    # Prioritize the provided file_path if it exists
    if filename:
        path = filename
    else:
        # Use the default URL based on the organism if no file_path is provided
        path = default_urls.get(organism.lower())
        if not path:
            raise ValueError("Invalid organism. Choose 'human' or 'mouse', or provide a custom file path.")

    try:
        # Read the CSV file
        df = pd.read_csv(path)

        # Check if required columns are present
        if 'symbol' not in df.columns or 'ensembl_id' not in df.columns:
            raise ValueError("CSV file must contain 'symbol' and 'ensembl_id' columns.")

        # Create and return the dictionary
        ensembl2symbol = dict(zip(df['ensembl_id'], df['symbol']))
        return ensembl2symbol

    except FileNotFoundError:
        print(f"File not found: {path}")
        return {}
    except pd.errors.EmptyDataError:
        print(f"The file is empty: {path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {}