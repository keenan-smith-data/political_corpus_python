import pandas
import scripts.data_pull as data_pull
import scripts.blosc_pickle as bi
import scripts.preprocess_text as ppt



if __name__ == "__main__":
    print("\nTokenizing Corpus\n")
    # Pull in Data
    full_corpus = data_pull.data_pull('combined_corpus')
    # Tokenizing text
    full_corpus['stemmed_token'] = full_corpus['text'].apply(ppt.preprocess_text)
    full_corpus['text_tokens'] = full_corpus['text'].apply(ppt.preprocess_text_no_stem)

    print("\nCorpus Tokenized, Stemmed, and Stopwords Removed\n")
    bi.blosc_pickle(full_corpus, "./data/tokenized_corpus.dat")
    print("\nProcess Completed\n")
