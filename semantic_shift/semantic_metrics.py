from WordVectorGroup import WordVectorGroup, WordFrequencyGroup
import pickle
import numpy as np
from itertools import combinations
from multiprocessing import Pool
from scipy.spatial.distance import cosine, euclidean
import re
import argparse


# Module variables
MIN_COMMON_VOCAB_SIZE = 10
f_distance = cosine


def cos_sq(u, v):
    """
    Returns the squared cosine distance between u and v.
    Args:
        u:
        v:

    Returns:

    """
    return cosine(u, v)**2


def get_common_mask(x, y):
    """
    Get a binary mask for common, non-zero entries between x and y.
    Args:
        x: matrix
        y: matrix
    Returns:
        mask: The binary mask for common non-zero vectors.
    """
    mask_x = ~np.all(x == 0, axis=1)  # Non-zero mask for x
    mask_y = ~np.all(y == 0, axis=1)  # Non-zero mask for y
    common_mask = (mask_x & mask_y)

    # Compute the jaccard index between the two vocabularies ( |x cap y|/|x cup y|)
    jaccard_index = (mask_x & mask_y).sum()/(mask_x | mask_y).sum()

    return common_mask, jaccard_index


def pairwise_distance(pair):
    """
    Given a pair of embedding matrices x and y, compute their cosine distance.
    Note: if the size of the common vocabulary is <= than `MIN_COMMON_VOCAB_SIZE`, the distance will be `np.nan`.
    Returns:
        d: The distance scalar between x and y.
    """
    x, y = pair

    common_mask, j_index = get_common_mask(x, y)

    if sum(common_mask) > MIN_COMMON_VOCAB_SIZE:
        d = (np.mean([f_distance(u, v) for u, v in zip(x[common_mask], y[common_mask])], dtype=float))
    else:
        d = np.nan

    return d


def pairwise_job(emb, i, j, d):
    """
    Runs a job to compute distances
    """

    x, y = emb[i], emb[j]
    d[i][j] = d[j][i] = pairwise_distance((x, y))
    return d[i][j]


def get_semantic_distance_matrix(wv_group, tgt_words, metric="cosine", workers=None,
                                 freq_group=None,
                                 n=None):
    """
    Given a WordVectorGroup `wv_group`, compute the pairwise distances between its embeddings.
    Embeddings in the group are aligned in a pairwise manner, then the distance is taken with respect to the set of
    target words in `tgt_words`.
    Args:
        wv_group: (WordVectorGroup) collection of WordEmbeddings
        tgt_words: (list) List of target words.
        metric: (str or callable) If str, one of {'cosine', 'euclidean', 'cosine-sim'}.
        If callable, it must take matrices `a`,`b` and return scalar `d` as the distance between the two matrices.
        workers: (int) Number of CPU cores to use for parallelism. If `None`, try to use all available cores.
        freq_group: (WordFrequencyGroup) (optional) If passed, words are selected based on frequency.
        n: (int) Choose the top n most frequent words from each source.
    Returns:
        d (n x n) matrix of distances where n is the number of sources in wv_group.

    """
    vocab_mask = wv_group.get_word_mask(tgt_words)  # This mask is used to get matrix rows for the target words

    wv_group.emb[:, ~vocab_mask] = 0

    print("Allocating space...")
    d = np.zeros((wv_group.group_size(), wv_group.group_size()))

    if callable(metric):
        fd = metric

    if freq_group:
        most_important_order = np.zeros((len(wv_group.labels), len(wv_group.idx_to_word)), dtype=int)
        most_important_mask = np.zeros((len(wv_group.labels), len(wv_group.idx_to_word)), dtype=bool)
        for i in range(len(most_important_order)):
            most_important_order[i] = np.argsort(freq_group.counts[i])[::-1]
            most_important_mask[i][most_important_order[i][:n]] = True  # Create a mask of most frequent words
            print("ORDER", most_important_order[i][:n])
            print(freq_group.counts[i][most_important_order[i][:n]])

    for i, e_i in enumerate(wv_group.emb):
        for j, e_j in enumerate(wv_group.emb[i+1:], start=i+1):
            if freq_group is not None:
                # n_indices = np.concatenate((most_important_order[i][:n], most_important_order[j][:n]))
                n_mask = most_important_mask[i] | most_important_mask[j]  # Add both binary masks
                x_i = wv_group.emb[i][n_mask]
                x_j = wv_group.emb[j][n_mask]
                print("WORDS", wv_group.idx_to_word[n_mask])
            else:
                x_i = wv_group.emb[i]
                x_j = wv_group.emb[j]

            # pd = pairwise_job(wv_group.emb, i, j, d)
            pd = pairwise_distance((x_i, x_j))
            d[i][j] = d[j][i] = pd
            print(wv_group.labels[i], "-", wv_group.labels[j], " dist", pd)
            print()

    print("> Done")

    return d


def get_frequency_aligned_embeddings(wv_group, freq_group, stopwords):
    """
    We have two objects `wv_group` and `freq_group` and we need to align both the sources (labels) and the words
    in them.
    To do that, we select the intersecting labels (usually wv_group.labels _in_ freq_group.labels).
    Then, we select the intersecting words and filter idx_to_word and embedding matrices of each.
    """
    print("Sources ", len(wv_group.labels), len(freq_group.labels))

    # We need to align the frequency matrix to the embedding matrix order
    s_indices = np.where(np.isin(freq_group.labels, wv_group.labels))[0]
    word_xsect = set.intersection(set(wv_group.idx_to_word), set(freq_group.idx_to_word))
    word_xsect = np.array([w for w in sorted(word_xsect) if re.match(r"[a-zA-Z]+", w)])
    word_xsect = np.array([w for w in word_xsect if w not in stopwords])
    print(len(word_xsect))

    # Get the indices of words in each group
    w_wv_indices = np.where(np.isin(wv_group.idx_to_word, word_xsect))[0]
    w_f_indices = np.where(np.isin(freq_group.idx_to_word, word_xsect))[0]

    # Filter lists of words
    wv_group.idx_to_word = wv_group.idx_to_word[w_wv_indices]
    freq_group.idx_to_word = freq_group.idx_to_word[w_f_indices]

    # Filter for labels
    freq_group.labels = freq_group.labels[s_indices]
    freq_group.counts = freq_group.counts[s_indices, :]

    # Filter matrices for words in common
    print(wv_group.emb.shape)
    wv_group.emb = wv_group.emb[:, w_wv_indices]
    print(wv_group.emb.shape)

    print(freq_group.counts.shape)
    freq_group.counts = freq_group.counts[:, w_f_indices]
    print(freq_group.counts.shape)

    wv_group.word_to_idx = {w: i for i, w in enumerate(wv_group.idx_to_word)}
    freq_group.word_to_idx = {w: i for i, w in enumerate(freq_group.idx_to_word)}

    return wv_group, freq_group


def main():
    input_path = "embeddings/source_word_embeddings.pickle"
    output_path = "embeddings/pairwise-distance-matrix--new.pickle"
    input_freq_path = "embeddings/frequencies.pickle"

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="No. of top freq words to consider for each source.")
    parser.add_argument("--f", type=float, help="Percentage of most frequent to consider.")
    parser.add_argument("--distance-function", dest="distance_function", choices=["cosine", "euclidean", "cos2"],
                        default="cosine",
                        help="Distance function used to measure semantic shift.")
    parser.add_argument("--min-vocab-size", dest="min_vocab_size", default=10, type=int,
                        help="Minimum vocabulary size required to consider semantic shift")

    args = parser.parse_args()

    if args.distance_function:
        global f_distance
        if args.distance_function == "euclidean":
            f_distance = euclidean
        elif args.distance_function == "cos2":
            f_distance = cos_sq

    if args.min_vocab_size:
        global MIN_COMMON_VOCAB_SIZE
        MIN_COMMON_VOCAB_SIZE = args.min_vocab_size

    with open(input_path, "rb") as fin:
        wv_group = pickle.load(fin)

    with open("../data/stopwords_english.txt") as fin:
        stopwords = set(map(lambda s: s.strip(), fin.readlines()))

    if args.n:
        with open(input_freq_path, "rb") as fin:
            freq_group = pickle.load(fin)
        wv_group, freq_group = get_frequency_aligned_embeddings(wv_group, freq_group, stopwords)
    else:
        freq_group = None

    print(wv_group.labels)

    tgt_words = [w for w in wv_group.idx_to_word if w not in stopwords]
    d = get_semantic_distance_matrix(wv_group, tgt_words, freq_group=freq_group,
                                     n=args.n)

    print(d)

    with open(output_path, "wb") as fout:
        pickle.dump((wv_group.labels, d), fout)

    return wv_group.labels, d


if __name__ == "__main__":
    main()
