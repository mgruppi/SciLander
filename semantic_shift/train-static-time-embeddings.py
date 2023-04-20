from gensim.models import Word2Vec
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import re
import os
from mapping import read_vocabulary
from WordVectorGroup import WordVectorSeries
import pickle
import argparse
from train_static_embeddings import process_corpus


def get_time_slice(con, t_start, t_end):
    """
    Selects all data from a NELA database connected via `con` with publication time `t`
    where `t_start` <= `t` < `t_end` (times are in unix epoch seconds).
    Args:
        con: sqlite3 connection to NELA database
        t_start: Unix epoch timestamp for start time
        t_end: Unix epoch timestamp for end time

    Returns:
        df - pandas.DataFrame containing the data for the given time slice
    """

    query = "SELECT * FROM newsdata WHERE published_utc >= %d AND published_utc < %d" % (int(t_start), int(t_end))
    df = pd.read_sql(query, con)
    return df


def process_time_corpus(documents, patterns, substrings):
    """
    Process a given collection of documents by cleaning and tokenizing each of them, combining them into a list
    of tokenized sentences.
    Args:
        documents: iterable or list(str) - Collection of documents to be processed.
        patterns: list of regex patterns to search in the documents.
        substrings: list of substrings to replace for each pattern in `patterns`.

    Returns: sentences - list(str) - A list containing the pre-processed and tokenized sentences from `documents`.
    """
    return process_corpus(documents, patterns, substrings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_delta", type=int, default=4, help="Time delta (in weeks)[default=4].")

    args = parser.parse_args()

    db_path = "../../data/collection/nela-covid-2020.db"
    labels_path = "../data/labels_all.csv"
    vocab_path = "../data/COVID_vocab.txt"

    w2v_params = {"vector_size": 100,
                  "window": 5,
                  "min_count": 10,
                  "workers": 64}

    # make output dir
    if not os.path.exists("embeddings/static/"):
        os.makedirs("embeddings/static/")

    path_out_rel = "embeddings/static/time_reliable.pickle"
    path_out_unr = "embeddings/static/time_unreliable.pickle"

    time_delta = args.time_delta  # Time delta used to split the data into periods (in weeks)
    con = sqlite3.connect(db_path)

    labels = pd.read_csv(labels_path)
    labels["class"] = (labels["questionable-source"] + labels["conspiracy-pseudoscience"]) > 0
    sources_rel = set(labels[labels["class"] is False]["source"].unique())
    sources_unr = set(labels[labels["class"] is True]["source"].unique())

    t_0 = datetime.strptime("2020-01-01", "%Y-%m-%d")
    t_f = datetime.strptime("2021-01-01", "%Y-%m-%d")

    t_current = t_0
    t_delta = timedelta(weeks=time_delta)

    # Read target vocabulary
    vocab = read_vocabulary(vocab_path, split=False)  # read vocabulary as a set().

    # Create regex from vocabulary
    patterns = list()
    sub_strings = list()
    for v in vocab:
        tokens = re.split("\W", v)
        r = re.compile(".".join(tokens))
        patterns.append(r)
        sub_strings.append("_".join(tokens))

    vocab_rel = list()
    emb_rel = list()
    vocab_unr = list()
    emb_unr = list()
    while t_current < t_f:
        print("- Period starting on %s" % t_current.strftime("%Y-%m-%d"))
        df = get_time_slice(con, t_current.strftime("%s"), (t_current+t_delta).strftime("%s"))

        df_rel = df[df["source"].isin(sources_rel)]
        df_unr = df[df["source"].isin(sources_unr)]

        t_current += t_delta  # Time step

        print("Pre-processing...")
        sents_rel = process_time_corpus(df_rel["content"], patterns, sub_strings)
        sents_unr = process_time_corpus(df_unr["content"], patterns, sub_strings)

        print("Training...")
        model_rel = Word2Vec(sentences=sents_rel, **w2v_params)

        vocab_rel.append(model_rel.wv.index_to_key)
        emb_rel.append(model_rel.wv.vectors)

        model_unr = Word2Vec(sentences=sents_unr, **w2v_params)

        vocab_unr.append(model_unr.wv.index_to_key)
        emb_unr.append(model_unr.wv.vectors)

    print("Creating Time Word Vectors")
    wv_rel = WordVectorSeries(vocab_rel, emb_rel)
    wv_unr = WordVectorSeries(vocab_unr, emb_unr)

    with open(path_out_rel, "wb") as fout:
        pickle.dump(wv_rel, fout)

    with open(path_out_unr, "wb") as fout:
        pickle.dump(wv_unr, fout)


if __name__ == "__main__":
    main()
