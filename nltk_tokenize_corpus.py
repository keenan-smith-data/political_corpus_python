def main():
    import pandas as pd
    import scripts.data_pull as data_pull
    import scripts.blosc_interface as bi
    import scripts.preprocess_text as ppt
    from nltk.stem import WordNetLemmatizer
    from alive_progress import alive_bar

    wordnet_lem = WordNetLemmatizer()

    with alive_bar(6) as bar:
        # Pull in Data
        full_corpus = data_pull.data_pull('combined_corpus')
        bar()
        
        # Tokenizing text
        full_corpus['stemmed_tokens'] = full_corpus['text'].apply(ppt.preprocess_text)
        bar()
        
        full_corpus['text_tokens'] = full_corpus['text'].apply(ppt.preprocess_text_no_stem)
        print("\nCorpus Tokenized, Stemmed, and Stopwords Removed\n")
        bar()
        
        print("\nRejoining Tokens with Commas\n")
        full_corpus['tokens'] = full_corpus["text_tokens"].apply(', '.join)
        bar()
        
        print("\nRejoining Stems with Commas\n")
        full_corpus['stems'] = full_corpus['stemmed_tokens'].apply(', '.join)
        bar()
        
        print("\nLemmatization with Commas\n")
        full_corpus['text_lem'] = full_corpus['tokens'].apply(wordnet_lem.lemmatize)
        bar()

        print("\nWork Done. Time to Compress and Store.\n")
        bi.blosc_pickle(full_corpus, "./data/tokenized_corpus.dat")
        print("\nProcess Completed\n")

if __name__ == "__main__":
    main()
