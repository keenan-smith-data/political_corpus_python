import pandas
import nltk
import scripts.data_pull as data_pull
import scripts.data_store as data_store

def stopword_removal(x):
    """A function for helping with stopword removal"""
    return [item for item in x if item not in project_stopwords]

if __name__ == "__main__":
    print("\nTokenizing Corpus\n")
    # Pull in Data
    full_corpus = data_pull.data_pull('combined_corpus')
    # Importing Stopwords
    from nltk.corpus import stopwords

    project_stopwords = stopwords.words("english")
    corpus_specific_stopwords = ['000', '10']
    project_stopwords.extend(corpus_specific_stopwords)
    # Tokenizing text
    full_corpus['text_token'] = full_corpus['text'].apply(nltk.tokenize.word_tokenize)
    print("\nCorpus Tokenized. Removing Stopwords Now\n")
    # Removing stopwords from Text
    full_corpus['text_token'] = full_corpus['text_token'].apply(stopword_removal)
    print("\nCorpus Tokenized and Stopwords Removed\n")
    data_store.data_store(full_corpus, "tokenized_corpus")
    print("\nProcess Completed\n")
