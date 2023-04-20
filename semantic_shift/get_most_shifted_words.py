from WordVectorGroup import WordVectorGroup
import pickle
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import re
import argparse


def read_vocabulary(path, return_tuples=False, split=True, stopwords={}, sep=" "):
    """
    Constructs a set of words given an input text file containing a given vocabulary.
    E.g., the CDC vocabulary on diseases can be used to load related words.
    If a vocabulary is given, the computations are carried out only on the words observed in the vocab.
    N-grams are converted to unigrams by splitting at spaces. E.g.: "Yellow Fever" is split into {"yellow", "fever"}
    Args:
        path: (string) Path to vocabulary file.
        return_tuples: (bool) if True, returns phrases as tuples of strings. E.g.: "hand washing"->('hand','washing').
        split: (bool) Split tokens in each line.
        stopwords: (set) Stopwords to remove from the list.
        sep (str): Separator to be used when creating n-grams.
    Returns:
        vocab: (set(str)) set of words/tuples read from the vocabulary (strings).
    """
    vocab = set()
    with open(path) as fin:
        for line in fin:
            if return_tuples:  # add tuples to the vocabulary
                vocab.add(tuple(line.lower().strip().split(" ")))
            elif split:  # each word as a vocabulary item
                tokens = line.lower().strip().split(" ")
                for t in tokens:
                    if t not in stopwords:
                        vocab.add(t)
            else:
                vocab.add(sep.join(line.strip().split(" ")))
    return vocab


def get_semantic_distances(a, b, d_func=cosine):
    """
    Given 2 embedding matrices a and b (row-aligned), compute the vector of semantic distances for each word in common.
    If a vector's norm is 0, the distance will be treated as NaN.
    Args:
        a: Embedding matrix a (n x d)
        b: Embedding matrix b (n x d)
        d_func: Distance function to apply. Must receive two arguments (u, v) and return a distance value.

    Returns:
        d - the vector of semantic distances (n)
    """

    d = np.zeros(len(a), dtype=float)
    range_i = range(len(a))
    for u, v, i in zip(a, b, range_i):
        # Either of the vectors have zero-norm
        if sum(u) == 0 or sum(v) == 0:
            d[i] = -np.infty
        else:
            d[i] = d_func(u, v)
    return d


def valid_word(w, stopwords, vocab):
    r = re.compile(r"\W+")
    is_valid = (r.match(w) is None) and (w not in stopwords) and (len(w) > 1)
    if vocab is not None:
        is_valid = is_valid and w in vocab
    return is_valid


def get_top_shifted_words(wv_group, a, b, stopwords, n=50, d_func=cosine,
                          vocab=None):
    a_idx = wv_group.label_to_idx[a]
    b_idx = wv_group.label_to_idx[b]
    d_ab = get_semantic_distances(wv_group.emb[a_idx], wv_group.emb[b_idx], d_func=d_func)
    most_shifted_order = np.argsort(d_ab)[::-1]
    top_words = [w for w in wv_group.idx_to_word[most_shifted_order] if valid_word(w, stopwords, vocab)]

    print(*top_words[:n], sep=",")

    return top_words[:n]


def collect_shifted_words(wv_group, a, b, stopwords, n=50, d_func=cosine, vocab=None):
    top_words = get_top_shifted_words(wv_group, a, b, stopwords, n, d_func=d_func, vocab=vocab)

    return top_words


if __name__ == "__main__":

    path_embeddings = "embeddings/cluster_word_embeddings.pickle"

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="No. of words to show")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean"], help="Metric")
    parser.add_argument("--vocab", type=str, default=None, help="Path to vocabulary terms")

    args = parser.parse_args()

    if args.metric == "euclidean":
        d_func = euclidean
    else:
        d_func = cosine

    if args.vocab:
        vocab = read_vocabulary(args.vocab, split=False)
    else:
        vocab = None

    with open(path_embeddings, "rb") as fin:
        wv_group = pickle.load(fin)

    print(wv_group.labels)

    with open("../data/stopwords_english.txt") as fin:
        stopwords = set(map(lambda s: s.strip(), fin.readlines()))

    a = "cluster_a"  # Hyperpartisan/right-wing conspiracy
    b = "cluster_b"  # Alternative medicine
    c = "cluster_c"  # Mainstream/center

    print("CLUSTER A to CLUSTER C")
    s = collect_shifted_words(wv_group, c, a, stopwords, args.n, d_func=d_func, vocab=vocab)
    print("MOST SHIFTED WORDS")
    print(*s)
    print("---" * 10)
    print("CLUSTER B TO CLUSTER C")
    s = collect_shifted_words(wv_group, c, b, stopwords, args.n, d_func=d_func, vocab=vocab)
    print(*s)



