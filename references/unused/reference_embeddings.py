"""
This file contains modules to run source-level reference embeddings.
"""
import argparse
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine


def train_embeddings(data, **kwargs):
    model = Word2Vec(sentences=data, **kwargs)
    return model


def load_input(path):
    """
    Reads file in path and returns sentences as list of tokens
    Note: if input file is too large, it is best to use LineSentence, without closing the file, and use the
            LineSentence object to stream data into a Word2Vec model.
    Return:
          sentences     - list, sentences as lists of tokens
    """
    with open(path) as fin:
        sentences = list()
        for line in fin:
            sent = line.strip().split(" ")
            sentences.append(sent)
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Path to save embeddings files to")
    parser.add_argument("--size", type=int, default=50, help="Size (dimensions) of embedded vectors")
    parser.add_argument("--window", type=int, default=10, help="Context window size")
    parser.add_argument("--min_count", type=int, default=10, help="Minimum count threshold")
    parser.add_argument("--workers", type=int, default=12, help="No. of worker threads.")

    args = parser.parse_args()

    data = load_input(args.input)

    params = dict()
    params["size"] = args.size
    params["workers"] = args.workers
    params["window"] = args.window
    params["min_count"] = args.min_count

    model = train_embeddings(data, **params)

    print(cosine(model.wv["thenewyorktimes"], model.wv["alternet"]))
    print(cosine(model.wv["thenewyorktimes"], model.wv["cnn"]))

    model.wv.save_word2vec_format(args.output)


if __name__ == "__main__":
    main()