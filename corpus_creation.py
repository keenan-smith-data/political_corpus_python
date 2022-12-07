# Importing Pandas
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
        df.to_pickle(data_path+data_filename+".pkl", compression={'method': 'gzip', 'compresslevel': 9, 'mtime': 1})


if __name__ is __main__:
    # Data Type Manipulation
    corpus_type_dict = {'text': 'string', 'art_title': 'string', 'art_author': 'string', 'art_topic': 'string', 'art_link': 'string', 'art_source': 'string'}
    # List of Column Names
    corpus_cols = ['text', 'art_title', 'art_author', 'art_date', 'art_topic', 'art_link', 'art_source']

    # Reading Raw Data into Pandas Dataframe from Pickle
    jacobin_text = data_pull("jacobin_text")
    brooking_text = data_pull("brooking_text")
    heritage_com_text = data_pull("heritage_com_text")
    heritage_rep_text = data_pull("heritage_rep_text")
    american_mind_features_text = data_pull("american_mind_features_text")
    american_mind_memos_text = data_pull("american_mind_memos_text")
    american_mind_salvos_text = data_pull("american_mind_salvos_text")
    # Joining the corpus into a single dataframe
    data_corpus = pd.concat([jacobin_text, brooking_text, heritage_com_text, heritage_rep_text, american_mind_features_text, american_mind_memos_text, american_mind_salvos_text])
    # Dropping Empty Rows
    data_corpus = data_corpus.dropna()
    # Correcting the Column Types
    data_corpus = data_corpus.astype(corpus_type_dict)
    data_corpus['art_date'] = pd.to_datetime(data_corpus['art_date'], yearfirst= True)
    # Group by and Apply to combined text corpus into a single row per article
    test_corpus = data_corpus.groupby(corpus_cols[1:])['text'].apply(' '.join).reset_index()


    data_store(test_corpus, data_filename= 'combined_corpus')