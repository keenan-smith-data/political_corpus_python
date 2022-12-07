# Importing Pandas
import pandas as pd
import scripts.data_pull as dp
import scripts.data_store as ds


if __name__ == "__main__":
    # Data Type Manipulation
    corpus_type_dict = {'text': 'string', 'art_title': 'string', 'art_author': 'string', 'art_topic': 'string', 'art_link': 'string', 'art_source': 'string'}
    # List of Column Names
    corpus_cols = ['text', 'art_title', 'art_author', 'art_date', 'art_topic', 'art_link', 'art_source']

    # Reading Raw Data into Pandas Dataframe from Pickle
    jacobin_text = dp.data_pull("jacobin_text")
    brooking_text = dp.data_pull("brooking_text")
    heritage_com_text = dp.data_pull("heritage_com_text")
    heritage_rep_text = dp.data_pull("heritage_rep_text")
    american_mind_features_text = dp.data_pull("american_mind_features_text")
    american_mind_memos_text = dp.data_pull("american_mind_memos_text")
    american_mind_salvos_text = dp.data_pull("american_mind_salvos_text")
    # Joining the corpus into a single dataframe
    data_corpus = pd.concat([jacobin_text, brooking_text, heritage_com_text, heritage_rep_text, american_mind_features_text, american_mind_memos_text, american_mind_salvos_text])
    # Dropping Empty Rows
    data_corpus = data_corpus.dropna()
    # Correcting the Column Types
    data_corpus = data_corpus.astype(corpus_type_dict)
    data_corpus['art_date'] = pd.to_datetime(data_corpus['art_date'], yearfirst= True)
    # Group by and Apply to combined text corpus into a single row per article
    test_corpus = data_corpus.groupby(corpus_cols[1:])['text'].apply(' '.join).reset_index()


    ds.data_store(test_corpus, data_filename= 'combined_corpus')