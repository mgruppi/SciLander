import numpy as np
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def pairwise_semantic_distances(a):
    """
    Given an array of semantic affinities a, return the pairwise absolute difference |ai - aj| i,j in a.
    Args:
        a: Array of semantic affinity

    Returns:
        x: The distance matrix from a
    """
    x = np.zeros((len(a), len(a)), dtype=float)
    for i, b in enumerate(a):
        for j, c in enumerate(a[i+1:], start=i+1):
            x[i][j] = x[j][i] = np.abs(b-c)
    return x


def generate_triplets(sources, tgt_words, affinity):
    """
    Generates triplets to train embeddings based on semantic shift.
    Samples triplets based on distance between affinity scores. Positive samples are pairs of sources with similar
    scores. Negative samples are uniformly sampled.
    Args:
        sources: list of source names (size n)
        tgt_words: list of target words (size m)
        affinity: affinity matrix (n x m affinity matrix)

    Returns:
        triplets (list(tuple)) - Training triplets with anchor, positive and negative samples.
    """
    triplets = list()

    dist_threshold = 0.01
    non_zero = np.where(~np.all(affinity == 0, axis=0))[0]

    for idx_word in non_zero:
        d_word = affinity[:, idx_word]  # Semantic shift for word `idx_word`
        x = pairwise_semantic_distances(d_word)

        for i in range(len(x)):
            for j in range(i+1, len(x[0])):
                if x[i][j] < dist_threshold:
                    neg_sample = np.random.choice(sources, size=1)[0]
                    t = (sources[i], sources[j], neg_sample)
                    triplets.append(t)

    return triplets


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity(x):
    return x


def squared_distance(x):
    return x**2


def generate_triplets_from_pairwise_distances(sources, distances, n_samples=100,
                                              f_apply=identity,
                                              allow_loops=False,
                                              uniform_negatives=False,
                                              restrict_positives=True,
                                              alpha=1):
    """
    Generate source triplets (a, p, n) where a is the anchor source, p is a positive sample (close to a), and
    n is a negative sample (distant from a).
    Args:
        sources: (list) List of source names.
        distances: (np.ndarray) Pairwise distance matrix.
        n_samples: (int) The number of walks to perform per source.
        f_apply: (callable) Function to apply to probability distribution.
        allow_loops: (bool) Allow triplets that pair the anchor to itself (Default: False).
        If False, triplets that contain two or more of the same source will be discarded.
        uniform_negatives: (bool) If True, negative samples are drawn from a uniform distribution of sources
        with distances greater than the mean distance.
        restrict_positives: (bool) If True, positive samples are restricted to a subset of sources (d less than mean).
        alpha: (float) No. of standard deviations to consider when drawing samples.
    Returns:
        triplets (list) - triplets of the form (a, p, n)
    """

    triplets = list()

    # distances = np.nan_to_num(distances, nan=2)  # Replace nan's with maximum distance (2 for cosine dist.)
    sources = np.array(sources)  # transform to np array

    for i, s in enumerate(sources):
        # Sample negatives
        d_mask = (~np.isnan(distances[i]))
        d_mask[i] = False

        if d_mask.sum() == 0:
            continue

        d_ = distances[i][d_mask]

        # Get the binary mask for negative and positive sample candidates
        negative_mask = d_ > 0
        positive_mask = d_ < d_.mean() - d_.std()*alpha

        if not restrict_positives:
            positive_mask = np.ones(len(d_), dtype=bool)

        x = f_apply(d_)

        x_positives = x[positive_mask]

        x_negatives = x[negative_mask]

        if x_positives.sum() == 0:  # No positives to sample
            continue
        if x_negatives.sum() == 0:  # No negatives to sample, but we can do a random sample
            continue
        else:
            if not uniform_negatives:
                p_neg = x_negatives/x_negatives.sum()
            else:
                p_neg = None

        x_positives = 1/x_positives
        p_pos = (x_positives/x_positives.sum())
        if p_pos.sum() < 1:  # Maybe we still need to normalize
            p_pos /= p_pos.sum()

        nearest_ = np.argsort(p_pos)
        print(s, sources[d_mask][positive_mask][nearest_][:10])
        # sns.histplot(p_pos)
        # plt.show()

        # print(x_positives)
        # print(d_.mean(), d_.std())

        samples_pos = np.random.choice(sources[d_mask][positive_mask], size=n_samples, p=p_pos)
        samples_neg = np.random.choice(sources[d_mask][negative_mask], size=n_samples, p=p_neg)

        # d_sort = np.argsort(x)
        # sources_filtered = sources[d_mask]
        # print(s)
        # print("+++ near", *zip(sources_filtered[d_sort[0:20]], x[d_sort[:20]]))
        # print("---far", *zip(sources_filtered[d_sort[-20:]], x[d_sort[-20:]]))

        t = tuple((s, pos, neg) for pos, neg in zip(samples_pos, samples_neg)
                  if (pos != neg != s) or allow_loops)  # Drop self-pairs if needed

        triplets.extend(t)

    # Sort triplets by positive sample then by anchor
    triplets = sorted(triplets, key=lambda t: t[1])
    triplets = sorted(triplets, key=lambda t: t[0])

    return triplets


def reduce_distance_matrices(d: np.ndarray, **kwargs):
    """
    Given a (n x n x |v|) distance tensor, reduce the |v| distance matrices into a single n x n one.
    Args:
        d: (n x n x |v|) Distance tensor.
        This contains the pairwise distance matrices for each of the |v| vocabulary terms.
        **kwargs: Keyword arguments.
    Returns:
        d: The reduce (n x n) distance matrix.
    """

    d_ = d.T  # Transpose d so we now have a |v| x n x n matrix for easier manipulation with NumPy
    # each element d_[i] is a n x n matrix so we can count the number of common and NaN elements across all sources
    # with a single matrix operation.
    common_terms = d_[~np.isnan(d_).sum(axis=0)]
    print(common_terms)
    print(common_terms.shape)


def main():

    parser = argparse.ArgumentParser(description="Generate semantic shift based triplets for source embeddings.")
    parser.add_argument("--input", type=str, default=None, help="Path to input distance matrix pickle.")
    parser.add_argument("--output", type=str, default=None, help="Path to output triplets.")
    parser.add_argument("--n", type=int, default=100, help="No. of samples to draw, per source.")
    parser.add_argument("--allow-loops", dest="allow_loops", action="store_true",
                        help="Allow triplets with self-loops.")
    parser.add_argument("--uniform-negatives", dest="uniform_negatives",
                        action="store_true", help="Sample negatives uniformly.")
    parser.add_argument("--alpha", type=float, default=2,
                        help="How many standard deviations from the mean to sample pairs.")

    args = parser.parse_args()

    path_labels = "../data/source_labels.csv"
    labels = pd.read_csv(path_labels)

    if args.input is None:
        path_in = "embeddings/pairwise-distance-matrix.pickle"
    else:
        path_in = args.input

    if args.output is None:
        path_out = "../data/triplets/semantic.csv"
    else:
        path_out = args.output

    path_typos = "../data/source_typos.csv"
    np.random.seed(42)

    # Retrieve sources and distance matrix
    # sources, d = sm.main()

    with open(path_in, "rb") as fin:
        sources, d = pickle.load(fin)

    # Fix and normalize source names
    with open(path_typos) as fin:
        typos = dict(map(lambda s: s.strip().split(",", 1), fin.readlines()))

    sources = np.array([typos[s] if s in typos else s for s in sources])
    sources_in_labels = np.array([typos[s] if s in typos else s for s in labels["source"]])
    source_mask = np.isin(sources, sources_in_labels)

    sources = sources[source_mask]
    d = d[source_mask]
    d = d[:, source_mask]

    triplets = generate_triplets_from_pairwise_distances(sources, d, args.n, allow_loops=args.allow_loops,
                                                         uniform_negatives=args.uniform_negatives,
                                                         alpha=args.alpha,
                                                         )

    with open(path_out, "w") as fout:
        fout.write("a,p,n\n")
        for a, p, n in triplets:
            fout.write("%s,%s,%s\n" % (a, p, n))


if __name__ == "__main__":
    main()
