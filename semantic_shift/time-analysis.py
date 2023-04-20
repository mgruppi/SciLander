import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import os
from collections import namedtuple


def get_time_distances(df, emb_reference):
    """
    Returns a vector with the distances from words in `df` to the reference embeddings in `emb_reference`.
    E.g.: df is the set of embeddings in reliable sources, `emb_reference` are cord-19 or wikipedia embeddings.
    Args:
        df: DataFrame input embeddings
        emb_reference: Reference embeddings (dict-like).

    Returns:
        t: (np.ndarray) Time-series of distances over 12 months.
    """
    t_series = dict()

    for idx, emb in zip(df.index, df):
        word = idx[0]
        month = int(idx[1])
        x = emb

        if word not in t_series:
            t_series[word] = np.zeros(12, dtype=np.float32)

        if word in emb_reference:
            t_series[word][month-1] = cosine(x, emb_reference[word])
    return t_series


def time_series_semantic(args):
    """
    Construct time series for semantic distance to cord-19 and make plots.
    Args:
        df_rel: Reliable embeddings
        df_unr: Unreliable embeddings
        cord_embs: Cord-19 embeddings

    Returns:

    """
    labels = pd.read_csv(args.labels)
    labels["class"] = (labels["questionable-source"] + labels["conspiracy-pseudoscience"]) > 0

    sources_rel = set(labels[labels["class"] == False]["source"].unique())
    sources_unr = set(labels[labels["class"] == True]["source"].unique())

    with open(args.cord, "rb") as fin:
        df_cord = pickle.load(fin)

    cord_embs = df_cord[["word", "emb"]].groupby(["word"]).apply(lambda x: np.mean(x["emb"], axis=0))
    cord_embs = dict(cord_embs)

    with open(args.nela, "rb") as fin:
        df_nela = pickle.load(fin)
        df_nela = df_nela.drop_duplicates(["source", "word", "sent"])
    df_nela["month"] = pd.to_datetime(df_nela["timestamp"], unit="s").dt.strftime("%m")

    df_rel = df_nela[df_nela["source"].isin(sources_rel)]
    df_unr = df_nela[df_nela["source"].isin(sources_unr)]
    time_emb_rel = df_rel.groupby(["word", "month"]).apply(lambda x: np.mean(x["emb"], axis=0))
    time_emb_unr = df_unr.groupby(["word", "month"]).apply(lambda x: np.mean(x["emb"], axis=0))

    t_series_rel = get_time_distances(time_emb_rel, cord_embs)
    t_series_unr = get_time_distances(time_emb_unr, cord_embs)

    common_words = set.intersection(set(t_series_rel.keys()), set(t_series_unr.keys()))

    print(common_words)

    months = np.arange(1, 13)

    if not os.path.exists("../results/time-series-semantic"):
        os.makedirs("../results/time-series-semantic")
    for word in common_words:
        fig, ax = plt.subplots()

        ax.plot(months, t_series_rel[word], label="Reliable")
        ax.plot(months, t_series_unr[word], label="Unreliable")

        ax.set_xlabel("Month")
        ax.set_ylabel("Semantic distance to Cord-19")
        ax.set_title(word)
        ax.legend()

        fig.savefig("../results/time-series-semantic/ts_semantic_%s.png" % word)


def get_frequency_time_series(df):
    """
    Return dictionary of time series per word
    Args:
        df:

    Returns:

    """
    t_series = dict()

    for idx, f in zip(df.index, df["relative_frequency"]):
        word = idx[0]
        month = int(idx[1])

        if word not in t_series:
            t_series[word] = np.zeros(12, dtype=np.float32)
        t_series[word][month-1] = f

    return t_series


def time_series_frequency(args):
    """
    Construct time series for word frequencies.

    Returns:

    """
    labels = pd.read_csv(args.labels)
    labels["class"] = (labels["questionable-source"] + labels["conspiracy-pseudoscience"]) > 0

    sources_rel = set(labels[labels["class"] == False]["source"].unique())
    sources_unr = set(labels[labels["class"] == True]["source"].unique())

    with open(args.frequency, "rb") as fin:
        df = pickle.load(fin)
        df["month"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.strftime("%m")

    df_rel = df[df["source"].isin(sources_rel)]
    df_unr = df[df["source"].isin(sources_unr)]

    common_words = set.intersection(set(df_rel["word"].unique()), set(df_unr["word"].unique()))

    ts_rel = df_rel[["word", "month", "relative_frequency"]].groupby(["word", "month"]).mean()
    ts_unr = df_unr[["word", "month", "relative_frequency"]].groupby(["word", "month"]).mean()

    t_series_rel = get_frequency_time_series(ts_rel)
    t_series_unr = get_frequency_time_series(ts_unr)

    if not os.path.exists("../results/time-series-frequency"):
        os.makedirs("../results/time-series-frequency")

    months = np.arange(1, 13)

    for word in common_words:
        fig, ax = plt.subplots()

        ax.plot(months, t_series_rel[word], label="Reliable")
        ax.plot(months, t_series_unr[word], label="Unreliable")

        ax.set_xlabel("Month")
        ax.set_ylabel("Relative term frequency")
        ax.set_title(word)
        ax.legend()

        fig.savefig("../results/time-series-frequency/ts_frequency_%s.png" % word)


def main():
    Arguments = namedtuple("Arguments", ["cord", "nela", "frequency", "labels"])
    path_cord = "embeddings/cord19_embeddings.df"
    path_nela = "embeddings/bert_embeddings.df"
    path_frequency = "embeddings/nela_frequency.pickle"
    path_labels = "../data/labels_all.csv"

    args = Arguments(path_cord, path_nela, path_frequency, path_labels)

    semantic = True
    frequency = True
    if semantic is True:
        time_series_semantic(args)
    if frequency is True:
        time_series_frequency(args)


if __name__ == "__main__":
    main()
