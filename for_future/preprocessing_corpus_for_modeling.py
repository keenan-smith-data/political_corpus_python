import pandas as pd
import numpy as np
import scripts.data_pull as dp
import scripts.preprocess_text as ppt
import scripts.blosc_interface as bi

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer=ppt.preprocess_text, max_features=200)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_vect = TfidfTransformer()     

if __name__ == "__main__":
    print("\n Starting Pre-Processing Script\n")
    print("\n Reading in Data\n")
    full_corpus = dp.data_pull("combined_corpus")
    print("\n Splitting Data\n")
    x_text = full_corpus.text
    y_bias = full_corpus.art_bias

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        x_text, y_bias,
        test_size=.2,
        random_state=42
    )

    
    print("\nPickling y_training data \n")
    bi.blosc_pickle(y_train, "./data/corpus_bias_train.dat")

    print("\nPickling y_training data \n")
    bi.blosc_pickle(y_test, "./data/corpus_bias_test.dat")
    
    from sklearn.feature_extraction.text import CountVectorizer
    print("\nVectorizing Data. This could take awhile\n")
    count_vect = CountVectorizer(analyzer=ppt.preprocess_text, max_features=200)

    print("\nVectorizing Data. This could take awhile\n") 
    # Count Vectorizing Training Set
    x_train_vect = count_vect.fit_transform(x_train)

    print("\nStoring Vectorizor Atrributes\n")
    bi.blosc_pickle(count_vect, "./data/count_vect.dat")
    
    # Count Vectorizing Testing Set
    x_test_vect = count_vect.fit_transform(x_test)

    print("\nPickling x_training data \n")

    bi.blosc_pickle(x_train_vect, "./data/corpus_text_vectorized_train.dat")

    print("\nPickling x_test data \n")
    
    bi.blosc_pickle(x_test_vect, "./data/corpus_text_vectorized_test.dat")

    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_vect = TfidfTransformer()

    x_train_tfidf = tfidf_vect.fit_transform(x_train_vect)

    print("\nStoring Vectorizor Atrributes\n")
    bi.blosc_pickle(tfidf_vect, "./data/tfidf_vect.dat")

    x_test_tfidf = tfidf_vect.fit_transform(x_test_vect)

    print("\nPickling x_training tfidf data \n")
    bi.blosc_pickle(x_train_tfidf, "./data/corpus_tfidf_vectorized_train.dat")

    print("\nPickling x_test tfidf data \n")
    bi.blosc_pickle(x_test_tfidf, "./data/corpus_tfidf_vectorized_test.dat")

    print("\nCongrats. You are done.\n")
