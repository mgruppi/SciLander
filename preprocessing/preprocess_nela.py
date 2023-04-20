import argparse
import sqlite3
from nltk.tokenize import word_tokenize, sent_tokenize


def tokenize_text(text):
    """
    Tokenize sentences and tokens in a given text.
    :param text: Input text.
    :return sentences: List of tokenized sentences.
    """
    tokens = word_tokenize(text)
    sentences = sent_tokenize(" ".join(tokens))
    return sentences


def tokenize_text_batch(texts):
    """
    Tokenize a batch of texts.
    :param texts: List of input texts.
    :return tokenized_texts: Texts as lists of tokens.
    """
    tokenized_texts = list()
    for text in texts:
        sentences = tokenized_texts(text)
        tokenized_texts.append(sentences)
    return tokenized_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input NELA database.")
    parser.add_argument("output", type=str, help="Output text file.")
    args = parser.parse_args()

    path_db = "../../data/nela/release/nela-gt-2020.db"
    con = sqlite3.connect(path_db)

    query = "SELECT id, content FROM newsdata limit 100"
    i = 30
    _id, text = con.cursor().execute(query).fetchall()[i]
    print(_id, text)
    sentences = tokenize_text(text)

    print(*sentences, sep="\n")


if __name__ == "__main__":
    main()
