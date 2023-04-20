import pickle
import pandas as pd
import argparse
import os
from collections import Counter
from multiprocessing import Pool
import re

from WordVectorGroup import WordFrequencyGroup
from mapping import read_vocabulary
from train_static_embeddings import read_duplicates_file, exclude_articles, get_patterns_and_substrings, process_corpus


def count_words(document):
    return Counter(document)


def get_word_counts(df, source, patterns, substrings, min_count=100):
    d = df[df["source"] == source]
    sentences = process_corpus(d["content"], patterns, substrings)
    counts = Counter()

    # for sent in sentences:
    #     counts += Counter(sent)

    with Pool(None) as p:
        count_pool = p.map(count_words, sentences)

    for ct in count_pool:
        counts += ct
    regex = re.compile(r"\w+")
    counts = {w: counts[w] for w in counts
              if counts[w] >= min_count
              and regex.match(w) is not None}
    return counts


def main():
    path_corpus = "corpus/corpus.df"
    vocab_path = "../data/COVID_vocab.txt"
    path_out = "embeddings/frequencies.pickle"

    parser = argparse.ArgumentParser()
    parser.add_argument("--duplicates", type=str, default=None, help="Path to article duplicates")
    args = parser.parse_args()
    duplicates_file = args.duplicates

    if duplicates_file:
        duplicates = read_duplicates_file(duplicates_file)
        print("Found", len(duplicates), "duplicate articles.")
    else:
        duplicates = set()

    exclude = {*duplicates}  # Exclude these article ids from training

    df = pd.read_pickle(path_corpus)
    df = exclude_articles(df, exclude)

    vocab = read_vocabulary(vocab_path, split=False)
    patterns, substrings = get_patterns_and_substrings(vocab)  # get substitution patterns and strings for n-grams

    word_counts = dict()
    sources = df["source"].unique()

    for src in sources:
        print(src)
        word_counts[src] = get_word_counts(df, src, patterns, substrings, min_count=100)

    frequencies = WordFrequencyGroup(word_counts, sources, min_source_count=10)

    with open(path_out, "wb") as fout:
        pickle.dump(frequencies, fout)

    with open(path_out, "rb") as fin:
        frequencies = pickle.load(fin)

    print(frequencies.counts)
    print(frequencies.labels)
    print(frequencies.idx_to_word)


if __name__ == "__main__":
    main()
