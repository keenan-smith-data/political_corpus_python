from sklearn.model_selection import train_test_split


def corpus_split(corpus, sample = .2, random_sauce = 42):
    x_text = corpus.text_lem
    y_bias = corpus.art_bias

    print("\nSplitting the Data into Train and Test")
    x_train, x_test, y_train, y_test = train_test_split(
    x_text, y_bias,
    test_size=.2,
    random_state=42
    )

    return x_train, x_test, y_train, y_test