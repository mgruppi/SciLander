"""
Performs mapping of words between two input word embeddings A and B
"""
import os
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from alignment import align
from WordVectors import WordVectors, intersection
import seaborn as sb
import matplotlib.pyplot as plt


def read_word_frequency(path):
    """
    Constructs a dictionary of word -> frequency given an input CSV file
    The file must not contain a header and must be structured as
    <word>,<count>,<frequency>
    Where <count> is the absolute count of <word> and <frequency> is its relative frequency
    Args:
        path: (string) Path to word frequency file

    Returns:
        freq: (dict(float)) dictionary mapping word to its frequency
    """
    if path is None:
        return dict()
    with open(path) as fin:
        freq = dict(map(lambda a: (a.strip().split(",")[0], float(a.strip().split(",")[2])), fin.readlines()))
    return freq


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=str, help="Path to embedding A (source).")
    parser.add_argument("b", type=str, help="Path to embedding B (target).")
    parser.add_argument("--freq_a", type=str, default=None, help="Path to word frequency (source).")
    parser.add_argument("--freq_b", type=str, default=None, help="Path to word frequency (target).")
    parser.add_argument("--vocab", type=str, default=None, help="Path to vocabulary file.")
    parser.add_argument("--k", type=int, default=1, help="Number of neighbors to consider when matching words.")
    parser.add_argument("--non_matching", action="store_true", help="Save only non-matching samples to output file.")
    parser.add_argument("--min_size", type=int, default=100, help="Minimum size of output.")

    args = parser.parse_args()

    freq_a = read_word_frequency(args.freq_a)
    freq_b = read_word_frequency(args.freq_b)

    if args.vocab:
        vocab = read_vocabulary(args.vocab)
    else:
        vocab = None

    norm = False  # normalize vectors

    wv_a = WordVectors(input_file=args.a, normalized=norm)
    wv_b = WordVectors(input_file=args.b, normalized=norm)
    wa, wb = intersection(wv_a, wv_b)
    sw = stopwords.words("english")

    sw = list(set.intersection(set(sw), set(wa.words)))

    print("Stop words: %d" % len(sw))
    print("Common vocab.: %d" % len(wa))

    # Check if minimum size is achieved.
    if len(wa) < args.min_size:
        return
    if vocab:
        valid_vocab_words = set.intersection(set(wa.words), vocab)
        if len(valid_vocab_words) < args.min_size:
            return

    print("Aligning embeddings...")
    wva, wvb, Q = align(wa, wb)
    wv_a.vectors = np.dot(wv_a.vectors, Q)

    # Fit NN in wv_b space to find mappings from wv_a to wv_b
    print('Mapping neighbors...')
    nbrs = NearestNeighbors(n_neighbors=args.k, n_jobs=12, metric="cosine").fit(wvb.vectors)
    distances, indices = nbrs.kneighbors(wva.vectors)

    a_name = os.path.basename(args.a)
    b_name = os.path.basename(args.b)
    if not os.path.exists("embeddings"):
        os.mkdir("embeddings")
    if not os.path.exists("embeddings/mapping"):
        os.mkdir("embeddings/mapping")
    path_result = "embeddings/mapping"
    path_out = path_result+"/%s-%s-k%d.csv" % (a_name, b_name, args.k)

    # Compute distribution of distances for matching and non-matching words
    d_matching = list()
    d_non_matching = list()
    words_non_matching = list()

    if vocab:
        wordlist = vocab
    else:
        wordlist = wva.words

    num_words = len(wordlist)
    for w in wordlist:
        if w not in wva.word_id:
            continue
        i = wva.word_id[w]
        for j, wt in enumerate(indices[i]):
            if w == wvb.words[wt]:  # if w is among the k neighbors, it is a match!
                d_matching.append(distances[i][j])
                break
        else:  # FOR-ELSE
            d_non_matching.append(distances[i][j])
            words_non_matching.append(w)

    print("-" * 20)
    print(a_name, "-", b_name)
    print("Matching: %d/%d (%.3f)" % (len(d_matching), num_words, len(d_matching)/num_words))
    print("Non-matching: %d/%d (%.3f)" % (len(d_non_matching), num_words, len(d_non_matching) / num_words))
    print("-" * 20)

    sb.kdeplot(d_matching, label="matching")
    sb.kdeplot(d_non_matching, label="non-matching")
    plt.xlabel("Distance")
    plt.legend()
    plt.title("%s-%s" % (a_name, b_name))
    plt.savefig(path_out.replace(".csv", ".png"))

    with open(path_out, "w") as fout:
        fout.write("source_word,mapping,distance,frequency_A, frequency_B\n")
        if args.non_matching:
            wordlist = words_non_matching
        elif vocab:
            wordlist = vocab
        else:
            wordlist = wva.words
        for w in wordlist:
            if w in wva.word_id:
                i = wva.word_id[w]
            else:
                continue
            fa = freq_a[w] if w in freq_a else 0

            for j, iw in enumerate(indices[i]):
                fb = freq_b[wvb.words[iw]] if wvb.words[iw] in freq_b else 0

                fout.write("%s,%s,%.4f,%.2e,%.2e" % (w, wvb.words[indices[i][j]], distances[i][j],
                                                     fa, fb))

                fout.write("\n")


if __name__ == "__main__":
    main()