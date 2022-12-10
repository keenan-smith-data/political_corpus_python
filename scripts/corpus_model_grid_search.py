import pandas as pd
import numpy as np
import scripts.blosc_interface as bi
import scripts.corpus_split as cs
from sklearn.model_selection import RandomizedSearchCV
from alive_progress import alive_bar

def corpus_model_grid_search(filename, pipe_line, param_grid, iter = 40,
    random_s = 42, threads = -1, folds = 5, verbo = 10):
    """A function for running a randomized grid on sklearn pipelines
    to determine best training parameters for various models

    Args:
        filename (string): a string determining the location and name
        of the grid search results
        pipe_line (sklearn.pipeline): a sklearn Pipeline object
        param_grid (dict): a dictionary for the parameters wishing to be 
        optimized
        iter (int, optional): number of random iterations. Defaults to 40.
        random_s (int, optional): a random seed for reproducibility. Defaults to 42.
        threads (int, optional): num of threads to run. Defaults to -1.
        folds (int, optional): amount of cross validation folds. Defaults to 5.
        verbo (int, optional): a status update on each grid run. Lower means less. Defaults to 10.
    """

    assert isinstance(param_grid, dict)
    with alive_bar(7) as bar:
        print("\nReading in the tokenized corpus")

        full_corpus = bi.blosc_read("./data/tokenized_corpus.dat")
        bar()

        x_train, x_test, y_train, y_test = cs.corpus_split(full_corpus)
        bar()

        print("\nEstablishing Pipeline")
        clf = pipe_line
        bar()

        print("\nSetting up Parameter Grid")
        param_grid = param_grid
        bar()

        search = RandomizedSearchCV(
                        estimator=clf,
                        param_distributions=param_grid,
                        n_iter=iter,
                        random_state=random_s,
                        n_jobs=threads,
                        cv=folds,
                        verbose=verbo,
        )

        print("\nTraining Dataset")
        search.fit(x_train, y_train)
        bar()

        print("\nGrid Search Complete. Outputting to Dataframe")
        search_res = pd.DataFrame(search.cv_results_)
        bar()

        print("\nWriting Grid Search Metrics to Disk")
        bi.blosc_pickle(search_res, filename)
        bar()

        print("Job Complete.")