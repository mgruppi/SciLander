from nela_features.nela_features import NELAFeatureExtractor
import pandas as pd
import numpy as np


def main():
    path_corpus = "../semantic_shift/corpus/corpus.df"
    path_output = "nela-features.csv"

    print("Reading data...")
    df_corpus = pd.read_pickle(path_corpus)
    print(df_corpus.columns)
    nela = NELAFeatureExtractor()

    sources = df_corpus["source"].unique()

    test = "TEST EXAMPLE"
    v, features = nela.extract_style(test)

    with open(path_output, "w") as fout:
        fout.write("source,%s\n" % ",".join(features))
        for src in sources:
            df_source = df_corpus[df_corpus["source"] == src]
            articles = df_source["title"] + "\n" + df_source["content"]
            print(src)
            src_vector = None
            vector_counts = 0
            for article in articles:
                try:
                    feature_vector, feature_names = nela.extract_style(article)
                    v = np.array(feature_vector)
                    vector_counts += 1
                    if src_vector is None:
                        src_vector = np.copy(v)
                    else:
                        src_vector += v
                except Exception as e:
                    print("NELA exception", e)

            src_vector /= vector_counts
            fout.write("%s,%s\n" % (src, ",".join(str(i) for i in src_vector)))


if __name__ == "__main__":
    main()