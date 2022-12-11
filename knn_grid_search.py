def main():
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import KNeighborsClassifier
    import scripts.corpus_model_grid_search as grid_search

    results_filename = "./data/knn_grid_search_result.dat"

    log_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ("svd", TruncatedSVD()),
            ('knn_clf', KNeighborsClassifier()), # this needs to be a different solver for LASSO
        ])

    log_param_grid = {
            "vect__ngram_range": [(1, 1), (1, 2)],
            "vect__max_features": [200, 300, 400],
            "svd__n_components": [5, 15, 30, 45, 60],
            "knn_clf__max_features": [2, 3, 4, 5, 6],
        }

    grid_search.corpus_model_grid_search(results_filename, log_clf, log_param_grid)



if __name__ == "__main__":

    main()
