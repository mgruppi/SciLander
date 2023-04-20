from WordVectorGroup import WordVectorGroup
from static_time_analysis import load_cord19
from mapping import read_vocabulary
from WordVectors import WordVectors
from gensim.models import Word2Vec
import pickle
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.linalg import orthogonal_procrustes
import argparse
import time


def read_labels(path):
    """
    Read a CSV file with source labels.
    Args:
        path: (str) Path to labels CSV.

    Returns:
        labels (dict): Mapping of source names (str) to class names (str).
    """
    labels = pd.read_csv(path)[["source", "bias"]]
    bias_unr = {"questionable-source", "conspiracy-pseudoscience"}
    labels["label"] = pd.Series([r if r in bias_unr else "reliable" for r in labels["bias"]])
    return labels


def load_word2vec_model(path):
    """
    Loads a Word2Vec model and returns the list of words and embedding matrix of that model.
    Args:
        path: (str or path-like) Path to binary word2vec model.

    Returns:
        idx_to_word (list), vectors (np.ndarray) - The list of words and their respective embeddings.
    """
    model = Word2Vec.load(path)

    return model.wv.index_to_key, model.wv.vectors


def load_data_and_preprocess(input_path, output_path, min_source_count):
    """
    Loads the embedding models for every source in `input_path` and combines them into a single WordVectorGroup
    object.

    Args:
        input_path: (str) Path to folder containing the source embedding models.
        output_path: Path to save the WordVectorGroup object
        min_source_count: (int) Minimum source count for terms.
    Returns:
        wv_group (WordVectorGroup) - Combined and aligned word embeddings.
    """
    vocabs = list()
    embeddings = list()
    labels = list()  # source names
    for root, dirs, files in os.walk(input_path):
        for f in sorted(files):
            words, vectors = load_word2vec_model(os.path.join(root, f))
            vocabs.append(words)
            embeddings.append(vectors)
            labels.append(f.split(".")[0])

    wv_group = WordVectorGroup(vocabs, embeddings, labels=labels, min_count=min_source_count)

    print(wv_group.vocab_size(), "terms /", wv_group.group_size(), " sources")

    # Load Cord-19 embeddings and get row-aligned matrix
    wv_anchor = load_cord19()
    common_vocab = sorted(set.intersection(set(wv_group.idx_to_word), set(wv_anchor.words)))
    print("Common vocab with Cord-19", len(common_vocab))

    # Make a row-aligned version of cord-19 embeddings with the same indexing as wv_group's embeddings
    anchors_emb = np.zeros((len(common_vocab), wv_anchor.dimension), dtype=float)
    anchors_index = np.array([wv_group.word_to_idx[w]
                              for w in common_vocab], dtype=int)  # Create an index of rows to use in alignment
    for i, term in enumerate(common_vocab):
        anchors_emb[i] = wv_anchor[term]

    wv_group.align_all(anchors_emb, anchors_index)

    # anchor = load_cord19()
    # align_to_anchor(wv_group, anchor)

    with open(output_path, "wb") as fout:
        pickle.dump(wv_group, fout)

    return wv_group


def get_transform(a, b):
    """
    Given two input matrices (both n x m), return the transformation matrix q* (m x m) such that
    dot(a, q*) solves min_q ||qa - b||² (Orthogonal Procrustes).
    Args:
        a: Input matrix (n x m)
        b: Input matrix (n x m)

    Returns:
        q: The solution to the Orthogonal Procrustes objective.
    """
    q, _ = orthogonal_procrustes(a, b)
    return q


def get_pairwise_distances(wv_group, tgt_words, metric="cosine",
                           by_word=False):
    """
    Given a WordVectorGroup `wv_group`, compute the pairwise distances between its embeddings.
    Embeddings in the group are aligned in a pairwise manner, then the distance is taken with respect to the set of
    target words in `tgt_words`.
    Args:
        wv_group: (WordVectorGroup) collection of WordEmbeddings
        tgt_words: (list) List of target words.
        metric: (str or callable) If str, one of {'cosine', 'euclidean', 'cosine-sim'}.
        If callable, it must take matrices `a`,`b` and return scalar `d` as the distance between the two matrices.
        by_word: (bool) If True, the output is a |v| x n x n matrix (one distance matrix per vocabulary term).
        If False, returns a single n x n distance matrix.
    Returns:
        d (n x n) matrix of distances where n is the number of sources in wv_group.

    """
    vocab_mask = wv_group.get_word_mask(tgt_words)  # This mask is used to get matrix rows for the target words

    if by_word:
        d = np.zeros((wv_group.group_size(), wv_group.group_size(), wv_group.vocab_size()), dtype=float)
    else:
        d = np.zeros((wv_group.group_size(), wv_group.group_size()))

    if type(metric) is str:
        if metric == "cosine":
            if not by_word:
                def fd(a, b):
                    d = np.array([cosine(x, y) for x, y in zip(a, b)])
                    return d
            else:
                def fd(a, b, common_tgt_indices):
                    d = np.empty(len(a))
                    d[:] = np.nan
                    d[common_tgt_indices] = np.array([cosine(u, v)
                                                      for u, v in
                                                      zip(a[common_tgt_indices], b[common_tgt_indices])])
                    return d

        elif metric == "euclidean":
            def fd(a, b):
                d = a - b
                return d
    elif callable(metric):
        fd = metric

    # Alignment step
    # anchor = wv_group.emb[0]  # align to first
    # print("Alignment...")
    # for i, e_i in enumerate(wv_group.emb):
    #     q = get_transform(e_i, anchor)
    #     wv_group.emb[i] = np.dot(e_i, q)

    for i, e_i in enumerate(wv_group.emb):
        mask_i = wv_group.get_non_zero_mask(i)
        for j, e_j in enumerate(wv_group.emb[i+1:], start=i+1):
            mask_j = wv_group.get_non_zero_mask(j)
            # Get words that are non-zero in both embeddings
            common_mask = (mask_i & mask_j)  # This mask gives the matrix rows that are common between i and j
            common_tgt_mask = common_mask & vocab_mask
            common_tgt_indices = np.where(common_tgt_mask)

            if not by_word:
                d_i = fd(e_i[common_tgt_mask], e_j[common_tgt_mask])
                d_i = d_i.mean()
            else:
                d_i = fd(e_i, e_j, common_tgt_indices)

            d[i][j] = d_i
            d[j][i] = d_i

            print(" -", wv_group.labels[i], " - ", wv_group.labels[j],
                  sum(common_mask), "vocab")
    return d


def get_distance_to_anchor(wv_group, anchor, tgt_words):

    x = np.zeros((len(wv_group.emb), len(tgt_words)))
    for i, e in enumerate(wv_group.emb):
        for j, w in enumerate(tgt_words):
            if w in wv_group.word_to_idx and w in anchor.word_id:
                x[i][j] = 1-cosine(e[wv_group.word_to_idx[w]], anchor.vectors[anchor.word_id[w]])

    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic(x, k=1, x0=0, L=1):
    return L / (1 + np.exp(-(k*(x-x0))))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Force recomputing alignment and distance matrix")
    parser.add_argument("--pairwise", action="store_true",
                        help="Compute pairwise distances instead of distance to reference.")
    parser.add_argument("--by-word", dest="by_word", action="store_true",
                        help="Output is a set of distance matrices compute for each word in the common vocab")
    parser.add_argument("--target-all", dest="target_all", action="store_true",
                        help="Use the entire common vocabulary as target words (excluding stopwords).")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"],
                        help="Distance/affinity metric to use.")
    parser.add_argument("--align", action="store_true", help="Align embedding groups and save it.")
    parser.add_argument("--min-source-count", dest="min_source_count", type=int, default=10,
                        help="Minimum source frequency required for a word to be included in the vocab.")
    parser.add_argument("--input", type=str, default=None, help="Input path")
    parser.add_argument("--output", type=str, default=None, help="Output path")

    args = parser.parse_args()

    input_path = "embeddings/source/"
    output_path = "embeddings/source_word_embeddings.pickle"

    if args.input:
        input_path = args.input
    if args.output:
        output_path = args.output

    if not os.path.exists(output_path) or args.overwrite:  # Do preprocessing, if needed
        print("Running load_data_and_preprocess")
        wv_group = load_data_and_preprocess(input_path, output_path, args.min_source_count)
    else:
        with open(output_path, "rb") as fin:
            wv_group = pickle.load(fin)

    # if args.target_all:
    #     tgt_vocab = [w for w in wv_group.idx_to_word if w not in stopwords]
    #
    # else:
    #     tgt_vocab = list(read_vocabulary(vocab_path, split=False))
    #
    # # Compute distance matrix, if needed
    # if not os.path.exists(distance_path) or args.overwrite:
    #     if args.pairwise:
    #         print("Running pairwise distances")
    #         print("By word", args.by_word)
    #         print("Target all words", args.target_all)
    #         d = get_pairwise_distances(wv_group, tgt_vocab, metric=args.metric,
    #                                    by_word=args.by_word)
    #     else:
    #         anchor = load_cord19()
    #         d = get_distance_to_anchor(wv_group, anchor, tgt_vocab)
    #     output = (wv_group.labels, tgt_vocab, d)
    #
    #     with open(distance_path, "wb") as fout:
    #         pickle.dump(output, fout)
    #
    # with open(distance_path, "rb") as fin:
    #     source_names, tgt_words, d = pickle.load(fin)
    # print(d)
    # print(d.shape)


if __name__ == "__main__":
    main()

