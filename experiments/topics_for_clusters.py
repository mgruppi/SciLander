import pandas as pd
import numpy as np
import spacy
from collections import Counter, defaultdict
import os
import argparse


def get_term_frequency(cluster_data, terms, key):
    """
    Gets the term frequency for `terms` in `cluster_data[key]`.
    Args:
        cluster_data: Dict of cluster data.
        terms: List of terms to retrieve from cluster_data
        key: The key for cluster_data ('entities', 'noun_chunks').

    Returns:
        np.array - matrix of term frequencies n x d

    """
    tf = np.zeros((len(terms), len(cluster_data)), dtype=float)
    for i, ent in enumerate(terms):
        for c_idx in cluster_data:
            c_idx = int(c_idx)
            if ent in cluster_data[c_idx][key]:
                tf[i][c_idx] = cluster_data[c_idx][key][ent]
    return tf


def get_inverse_document_frequency(tf):
    """
    Returns a matrix of inverse document frequency.
    Args:
        tf: term-frequency matrix
        terms: List of terms

    Returns:
        np.array idf (size n)
    """
    idf = np.zeros((len(tf)), dtype=float)
    n_docs = tf.shape[1]
    for i, term in enumerate(tf):
        idf[i] = n_docs/(sum(tf[i] > 0))
    idf = np.log(idf)
    return idf


def get_tf_idf(tf, idf):
    tfidf = np.zeros(tf.shape, dtype=float)

    for i, tf_i in enumerate(tf):
        tfidf[i] = tf[i] * idf[i]
    return tfidf


def main():
    path_corpus = "../semantic_shift/corpus/corpus.df"
    path_clusters = "../results/cluster_cores.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-term-freq", dest="min_term_freq", type=int, default=5, help="Minimum term freq.")
    parser.add_argument("--n-samples", dest="n_samples", type=int, default=1000, help="No. of titles to sample")
    parser.add_argument("--input", type=str, default=None, help="Input file")
    parser.add_argument("--n-words", dest="n_words", type=int, default=10, help="No. of words to display")
    parser.add_argument("--spacy-model", dest="spacy_model", default="en_core_web_lg", help="Spacy model to use.")
    parser.add_argument("--force", action="store_true", help="Force recomputing topics")
    parser.add_argument("--label", default=None, type=str, help="Label to filter for (e.g. PERSON for entities)")

    args = parser.parse_args()

    output_path = "../results/topics"

    if args.input is None:
        path_nchunks = os.path.join(output_path, "noun_chunks.csv")
        path_entities = os.path.join(output_path, "entities.csv")
        path_lemmas = os.path.join(output_path, "lemmas.csv")
        path_input = path_nchunks
    else:
        path_input = os.path.join(output_path, args.input)

    clusters = pd.read_csv(path_clusters)
    if not os.path.exists(path_input) or args.force:
        try:
            os.makedirs(output_path)
        except FileExistsError as e:
            print(e, "exists")

        corpus = pd.read_pickle(path_corpus)

        article_cluster = corpus.join(clusters.set_index("source"), on="source", how="left")
        article_cluster = article_cluster.dropna(axis=0, subset=["cluster"])

        n_samples = args.n_samples

        print(article_cluster)
        print(article_cluster.columns)

        nlp = spacy.load(args.spacy_model)
        df_noun_chunks = {"noun_chunk": list(), "root": list(), "root_tag_": list(),
                          "source": list(), "cluster": list()}

        df_entities = {"entity": list(), "label_": list(), "source": list(), "cluster": list()}
        df_lemmas = {"lemma": list(), "label_": list(), "source": list(), "cluster": list()}

        # Extract topics for each cluster -- once it ran once, there is no need to run it again
        for idx in article_cluster["cluster"].unique():
            if idx == -1:  # Skip noisy samples
                continue
            print("|= Cluster", idx)
            data_c = article_cluster[article_cluster["cluster"] == idx]
            data_c = data_c.sample(n=n_samples, replace=True, random_state=42)
            print(data_c["source"].unique())
            titles = list(nlp.pipe(data_c["title"], n_process=32))

            for doc, src in zip(titles, data_c["source"]):
                for nc in doc.noun_chunks:
                    if nc.root.is_stop:
                        continue
                    df_noun_chunks["noun_chunk"].append(nc.lemma_.lower())
                    df_noun_chunks["root"].append(nc.root.lemma_.lower())
                    df_noun_chunks["root_tag_"].append(nc.root.tag_)
                    df_noun_chunks["source"].append(src)
                    df_noun_chunks["cluster"].append(int(idx))

                for ent in doc.ents:
                    df_entities["entity"].append(ent.text)
                    df_entities["label_"].append(ent.label_)
                    df_entities["source"].append(src)
                    df_entities["cluster"].append(int(idx))

                for token in doc:
                    if token.is_stop:
                        continue
                    df_lemmas["lemma"].append(token.lemma_)
                    df_lemmas["label_"].append(token.tag_)
                    df_lemmas["source"].append(src)
                    df_lemmas["cluster"].append(int(idx))

        df_noun_chunks = pd.DataFrame(df_noun_chunks)
        df_entities = pd.DataFrame(df_entities)
        df_lemmas = pd.DataFrame(df_lemmas)

        columns = ["cluster", "term", "root.lemma_", "label_", "term_count", "term_source_freq"]
        data = list()
        for idx in df_noun_chunks["cluster"].unique():
            df_ = df_noun_chunks[df_noun_chunks["cluster"] == idx]  # df_ from cluster
            print(df_["source"].value_counts())
            for nchunk in df_["noun_chunk"].unique():
                df_nchunk = df_[df_["noun_chunk"] == nchunk]  # df_ from cluster and noun chunk
                # Frequency over all sources in this cluster
                root_lemma_ = list(df_nchunk["root"].unique())[0]
                root_tag = list(df_nchunk["root_tag_"].unique())[0]
                src_fq = df_nchunk["source"].nunique()/df_["source"].nunique()
                term_fq = len(df_nchunk["source"])
                data_row = (idx, nchunk, root_lemma_, root_tag, term_fq, src_fq)
                data.append(data_row)

        df_noun_chunks = pd.DataFrame(data, columns=columns)
        df_noun_chunks.to_csv(path_nchunks, index=None)

        columns = ["cluster", "term", "label_", "term_count", "term_source_freq"]
        data = list()
        for idx in df_entities["cluster"].unique():
            df_ = df_entities[df_entities["cluster"] == idx]
            for ent in df_["entity"].unique():
                df_ent = df_[df_["entity"] == ent]
                src_fq = df_ent["source"].nunique()/df_["source"].nunique()
                label_ = list(df_ent["label_"].unique())[0]
                term_fq = len(df_ent["source"])
                data_row = (idx, ent, label_, term_fq, src_fq)
                data.append(data_row)
        df_entities = pd.DataFrame(data, columns=columns)
        df_entities.to_csv(path_entities)

        columns = ["cluster", "term", "label_", "term_count", "term_source_freq"]
        data = list()
        for idx in df_lemmas["cluster"].unique():
            df_ = df_lemmas[df_lemmas["cluster"] == idx]
            for lemma in df_["lemma"].unique():
                df_lemma = df_[df_["lemma"] == lemma]
                label_ = list(df_lemma["label_"].unique())[0]
                src_fq = df_lemma["source"].nunique()/df_["source"].nunique()
                term_fq = len(df_lemma["source"])
                data_row = (idx, lemma, label_, term_fq, src_fq)
                data.append(data_row)
        df_lemmas = pd.DataFrame(data, columns=columns)
        df_lemmas.to_csv(path_lemmas)

    df = pd.read_csv(path_input)

    df = df[df["term_count"] > args.min_term_freq]

    if args.label:
        if "label_" in df.columns:
            df = df[df["label_"] == args.label]

    print(df)

    idx_to_term = np.array(sorted([str(nc) for nc in df["term"].unique()]))
    term_to_idx = {n: i for i, n in enumerate(idx_to_term)}

    tf = np.zeros((len(idx_to_term), df["cluster"].nunique()))
    tsf = np.zeros((len(idx_to_term), df["cluster"].nunique()))

    for nc, c_idx, tc, tsc in zip(df["term"], df["cluster"], df["term_count"], df["term_source_freq"]):
        if nc not in term_to_idx:
            continue
        tf[term_to_idx[nc]][c_idx] = tc
        tsf[term_to_idx[nc]][c_idx] = tsc

    # Normalize term-frequency
    # tf = tf / tf.sum(axis=0)
    tf = np.log(1+tf)

    idf = get_inverse_document_frequency(tf)

    tf_idf = get_tf_idf(tf, idf)

    tsf_tf_idf = np.multiply(tsf, tf_idf)

    for c in range(tf_idf.shape[1]):
        top_tf_idf = np.argsort(tf_idf[:, c])[::-1]
        top_tf = np.argsort(tf[:, c])[::-1]
        top_tsf = np.argsort(tsf_tf_idf[:, c])[::-1]

        print("====== CLUSTER", c)
        print(clusters[clusters["cluster"] == c]["source"].to_numpy())
        print(" %30s | %30s | %30s" % ("TF", "TF-IDF", "TF-TSF-IDF"))
        print("---" * 100)
        for i in range(args.n_words):
            print("%30s | %30s | %30s"
                  % (idx_to_term[top_tf[i]], idx_to_term[top_tf_idf[i]], idx_to_term[top_tsf[i]]))


if __name__ == "__main__":
    main()
