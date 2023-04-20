import pandas as pd
from knn_classification import read_labels, run_fold_cross_val, run_leave_one_out_cross_val
import numpy as np
import matplotlib.pyplot as plt


def read_features(path):
    df = pd.read_csv(path)
    features = df.loc[:, df.columns != "source"].to_numpy()
    index_to_source = np.array((df["source"]))

    return features, index_to_source


if __name__ == "__main__":

    path_features = "../text-baselines/nela-features.csv"
    path_labels = "../data/labels_all.csv"

    features, index_to_source = read_features(path_features)
    labels = read_labels(path_labels, index_to_source)

    label_dict = {s: y for s, y in zip(labels["source"], labels["class"])}

    index_mask = np.isin(index_to_source, labels["source"])

    index_to_source = index_to_source[index_mask]
    features = features[index_mask]

    index_to_label = np.fromiter((label_dict[s] for s in index_to_source), dtype=int)

    # Drop NaN from features
    features = features[:, ~np.isnan(features).any(axis=0)]

    k_list = range(1, 21)
    k_acc_mean = list()
    k_f1_mean = list()
    metric = "cosine"
    cv = 20
    for k in k_list:
        k_acc, k_f1 = run_fold_cross_val(features, index_to_label, k, metric, cv)
        k_acc_mean.append(k_acc.mean())
        k_f1_mean.append(k_f1.mean())

    plt.plot(range(len(k_acc_mean)), k_acc_mean, label="Test Accuracy")
    plt.plot(range(len(k_f1_mean)), k_f1_mean, label="Test F1")
    plt.ylabel("Score")
    plt.xlabel("K")
    plt.legend()
    plt.show()
