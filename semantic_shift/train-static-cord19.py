from gensim.models import Word2Vec
from cord19_utils import read_articles_from_path, extract_body_text
import argparse
from nltk.tokenize import word_tokenize, sent_tokenize
from mapping import read_vocabulary
import re
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to Cord-19")
    args = parser.parse_args()
    vocab_path = "../data/COVID_vocab.txt"

    if not os.path.exists("embeddings/static/"):
        os.makedirs("embeddings/static")

    # Read target vocabulary
    vocab = read_vocabulary(vocab_path, split=False)  # read vocabulary as a set().

    w2v_params = {
        "vector_size": 100,
        "window": 5,
        "min_count": 10,
        "workers": 64
    }

    # Create regex from vocabulary
    patterns = list()
    substrings = list()
    for v in vocab:
        tokens = re.split("\W", v)
        r = re.compile(".".join(tokens))
        patterns.append(r)
        substrings.append("_".join(tokens))

    sentences = list()
    for data in read_articles_from_path(args.path):
        document = extract_body_text(data)
        _doc = document.lower()
        for pattern, subs in zip(patterns, substrings):
            _doc = pattern.sub(subs, _doc.lower())

        doc_sents = sent_tokenize(_doc)
        for i, sent in enumerate(doc_sents):
            sentences.append([t for t in word_tokenize(sent)])

    model = Word2Vec(sentences=sentences, **w2v_params)
    model.save("embeddings/static/cord19.model")


if __name__ == "__main__":
    main()