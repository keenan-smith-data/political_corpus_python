import pandas as pd

def data_store(df, data_filename, data_path = "./data/", data_type = "pickle"):
    """A function for storing a Pandas Dataframe as a csv or pickle

    Args:
        df (pd.DataFrame): The pandas dataframe to be stored
        data_filename (str): the desired filename for the df to be stored
        data_path (str, optional): the top level storage directory. Defaults to "./data/".
        data_type (str, optional): the type of data store desired. Defaults to "pickle".
    """
    # Assertions to ensure Data Types are correct
    assert isinstance(df, pd.DataFrame), f"df must be a Pandas DataFrame object"
    assert isinstance(data_filename, str), f"Filename must be a string"
    assert data_type == "pickle" or "csv", f"data_type must be either a pickle or csv"
    # if statement to select correct data storage function
    if data_type == "csv":
        df.to_csv(data_path+data_filename+".csv")
    elif data_type == "pickle":
        # noinspection SpellCheckingInspection
        df.to_pickle(data_path+data_filename+".pkl", compression={'method': 'gzip', 'compresslevel': 9, 'mtime': 1})