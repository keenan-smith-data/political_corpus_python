from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation

porter_stemmer = PorterStemmer()
project_stopwords = set(stopwords.words("english"))
corpus_specific_stopwords = ['000', '10', '19', "''", "'s", "``", "'", '"', "-"]
project_stopwords.union(set(corpus_specific_stopwords))
project_stopwords.union(set(punctuation))

def preprocess_text(str_input):
    """A Function for Processing text data
    This function filters out punctuation, stopwords, and tokenizes text
    it outputs a stemmed word

    Args:
        str_input (string): a string of text to be processed

    Returns:
        list: a list of stemmed processed words
    """
    return [porter_stemmer.stem(word) for word in tqdm(word_tokenize(str_input)) if word not in project_stopwords and not word.isdigit()]

def preprocess_text_no_stem(str_input):
    """A Function for Processing text data
    This function filters out punctuation, stopwords, and tokenizes text
    it outputs a stemmed word

    Args:
        str_input (string): a string of text to be processed

    Returns:
        list: a list of stemmed processed words
    """
    return [word for word in tqdm(word_tokenize(str_input)) if word not in project_stopwords and not word.isdigit()]