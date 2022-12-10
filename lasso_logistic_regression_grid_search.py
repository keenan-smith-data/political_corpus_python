import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from alive_progress import alive_bar
import scripts.blosc_interface as bi

if __name__ == "__main__":


    with alive_bar(7) as bar:
        print("\nReading in the tokenized corpus")
        full_corpus = bi.blosc_read("./data/tokenized_corpus.dat")
        bar()
        x_text = full_corpus.text_lem
        y_bias = full_corpus.art_bias
        print("\nSplitting the Data into Train and Test")
        x_train, x_test, y_train, y_test = train_test_split(
            x_text, y_bias,
            test_size=.2,
            random_state=42
        )
        bar()
        print("\nEstablishing Pipeline")
        log_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ("svd", TruncatedSVD()),
            ('log_clf', LogisticRegression(penalty="l1")),
        ])
        bar()
        print("\nSetting up Parameter Grid")
        log_param_grid = {
            "vect__ngram_range": [(1, 1), (1, 2)],
            "vect__max_features": [200, 300, 400],
            "svd__n_components": [5, 15, 30, 45, 60],
            "log_clf__C": np.logspace(-4, 4, 4),
        }
        bar()
        log_search = GridSearchCV(log_clf, log_param_grid, cv=5, n_jobs=-1)
        print("\nTraining Dataset")
        log_search.fit(x_train, y_train)
        bar()
        print("\nGrid Search Complete. Outputting to Dataframe")
        log_grid_search_res = pd.DataFrame(log_search.cv_results_)
        bar()
        print("\nWriting Grid Search Metrics to Disk")
        bi.blosc_pickle(log_grid_search_res, "./data/log_grid_search_result.dat")
        bar()
        print("Job Complete.")