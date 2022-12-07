import pandas as pd

def data_pull(data_filename, data_path = "./data/", data_type = 'pickle'):
    """A function for pulling in a csv or pickle for analysis

    Args:
        data_filename (str): A string of text that is just the file name without the extension
        data_path (str, optional): the top level directory. Defaults to "./data/".
        data_type (str, optional): the type of data store desired. Defaults to 'pickle'.

    Returns:
        pd.DataFrame: A Pandas DataFrame
    """
    # Assertions to ensure Data Types are correct
    assert isinstance(data_filename, str), f"Filename must be a string"
    assert data_type == "pickle" or "csv", f"data_type must be either a pickle or csv"
    # if statement to select correct data storage function
    if data_type == "csv":
        return pd.read_csv(data_path + data_filename + ".csv")
    elif data_type == "pickle":
        return pd.read_pickle(data_path + data_filename + ".pkl", compression={'method': 'gzip', 'compresslevel': 9, 'mtime': 1})