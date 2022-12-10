# Importing Pandas
import pandas as pd
import scripts.data_pull as dp
import scripts.data_store as ds
import scripts.bias_assignment as bias


if __name__ == "__main__":
    print("\nCreating the Combined Corpus\n")
    # Data Type Manipulation
    corpus_type_dict = {'text': 'string', 'art_title': 'string', 'art_author': 'string', 'art_topic': 'string', 'art_link': 'string', 'art_source': 'string'}
    # List of Column Names
    corpus_cols = ['text', 'art_title', 'art_author', 'art_date', 'art_topic', 'art_link', 'art_source']
    print("\nReading in the Separate Corpus\n")
    # Reading Raw Data into Pandas Dataframe from Pickle
    jacobin_text = dp.data_pull("jacobin_text")
    print("\nRead in the Jacobin Corpus\n")
    brooking_text = dp.data_pull("brooking_text")
    print("\nRead in the Brookings Institute Corpus\n")
    heritage_com_text = dp.data_pull("heritage_com_text")
    heritage_rep_text = dp.data_pull("heritage_rep_text")
    print("\nRead in the Heritage Foundation Corpus\n")
    american_mind_features_text = dp.data_pull("american_mind_features_text")
    american_mind_memos_text = dp.data_pull("american_mind_memos_text")
    american_mind_salvos_text = dp.data_pull("american_mind_salvos_text")
    print("\nRead in the American Mind Corpus\n")
    # Joining the corpus into a single dataframe
    data_corpus = pd.concat([jacobin_text, brooking_text, heritage_com_text, heritage_rep_text, american_mind_features_text, american_mind_memos_text, american_mind_salvos_text])
    print("\nData Combined\n")
    print("\nReadying the Corpus Now\n")
    # Dropping Empty Rows
    data_corpus = data_corpus.dropna()
    # Correcting the Column Types
    data_corpus = data_corpus.astype(corpus_type_dict)
    data_corpus['art_date'] = pd.to_datetime(data_corpus['art_date'], yearfirst= True)
    # Group by and Apply to combined text corpus into a single row per article
    full_corpus = data_corpus.groupby(corpus_cols[1:])['text'].apply(' '.join).reset_index()
    # Text Cleaning and Bias Assignment
    full_corpus['text'] = full_corpus['text'].astype(str).str.lower()
    full_corpus['art_topic'] = full_corpus['art_topic'].str.lower()
    full_corpus['art_bias'] = full_corpus["art_source"].apply(bias.bias_assignment)
    # Removing the only non-English row
    full_corpus = full_corpus.drop(21757)
    # Store Tailored Data data
    ds.data_store(full_corpus, data_filename='combined_corpus')
    print("Corpus Created")
