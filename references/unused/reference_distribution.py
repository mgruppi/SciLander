"""
Compute the distribution of references with respect to source and source quality
Requires a urls_source_label-type in the following CSV format:
<url>,<source_name>,<quality_score>
"""

import argparse
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("input", type=str, help="Path to input CSV")
    # args = parser.parse_args()
    f_labels = "data/url_source_labels.csv"
    f_counts = "data/source_articles.csv"

    with open(f_counts) as fin:
        fin.readline()  # remove header
        num_articles = dict(map(lambda ls: ls.strip().split(",", 1), fin.readlines()))

    df = pd.read_csv(f_labels, header=None)
    print(df)

    scores = [0, 1, 2]
    for s in scores:
        df_ = df[df[2] == s]
        print(len(df_))
        d = df_[1].value_counts()
        print(d.values)
        print(d.index)
        sb.distplot(d.values, label=s)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()