import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def read_labels(path):
    labels = dict()
    with open(path) as fin:
        fin.readline()  # read header
        for line in fin:
            row = line.strip().split(",")
            source = row[0]
            label = int(row[5] == "1" or row[6] == "1")
            labels[source] = label
    return labels


def get_embedding_matrix(df, word):
    """
    Given an input DataFrame and a word, returns the corresponding embedding matrix x of word vectors of `word`.
    """
    x_word = df[df["word"] == word]["emb"]
    x = np.array([xi for xi in x_word])

    return x


def sample_sources(df, word, n_samples=20):
    d = df[df["word"] == word]
    d = d.groupby("source").apply(lambda s: s.sample(min(n_samples, len(s)), random_state=0))
    return d


def get_similar_samples(df, word, _index=20):
    d = sample_sources(df, word)
    d = d.drop_duplicates("sent")
    d["sent"] = d["sent"].str.replace("\n", " ")
    x = get_embedding_matrix(d, word)
    print(x.shape)
    m = pairwise_distances(x, metric="cosine")

    indices = np.argsort(m[_index])
    distances = m[_index]

    indices_high = indices[:len(indices)//2]
    indices_low = indices[len(indices)//2+1:]

    dates = pd.to_datetime(d["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    print(d["source"][_index], dates[_index], d["sent"][_index], sep=" | ")
    print("---")

    # Get index 1 because 0 will be the same sentence (closest one).
    print("+", d["source"][indices_high[1]], dates[indices_high[1]],
          distances[indices_high[1]], d["sent"][indices_high[1]], sep=" | ")
    print("+", d["source"][indices_high[2]], dates[indices_high[2]],
          distances[indices_high[2]], d["sent"][indices_high[2]], sep=" | ")
    print("-", d["source"][indices_low[-1]], dates[indices_low[-1]],
          distances[indices_low[-1]], d["sent"][indices_low[-1]], sep=" | ")
    print("-", d["source"][indices_low[-2]], dates[indices_low[-2]],
          distances[indices_low[-2]], d["sent"][indices_low[-2]], sep=" | ")

    print("===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to combined embedding pickle.")
    parser.add_argument("--target", type=str, default="pandemic", help="Target word to get sentences for.")
    parser.add_argument("--index", type=int, default=1, help="Index to sample from")

    # labels_file = "../data/labels_all.csv"
    # labels = read_labels(labels_file)

    args = parser.parse_args()

    w = args.target
    _idx = args.index

    with open(args.input, "rb") as fin:
        data = pickle.load(fin)

    df = pd.DataFrame(data)
    print(df["word"].unique())
    # vocab = ["pandemic"]
    #
    # w = vocab[0]
    print(" --", w)
    get_similar_samples(df, w, _index=_idx)
    # get_similar_samples(df, w, _index=10)


if __name__ == "__main__":
    main()
