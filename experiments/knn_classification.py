import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import cross_val_score, cross_validate, LeaveOneOut

import seaborn as sns
import matplotlib.pyplot as plt


def read_source_embeddings(path):
    """
    Reads an embedding model in `path` and returns the data.
    Args:
        path: Path to embeddings model (Pandas DataFrame).
    Returns:
        index_to_source, source_to_index, embeddings: Respectively, the mapping of index to source name,
        the mapping of source name to index and the `embeddings` matrix of the sources.
    """
    with open(path, "rb") as fin:
        data = pickle.load(fin)
    source_to_index = {s: i for i, s in enumerate(data["source"])}
    index_to_source = np.array(list(data["source"]))

    # # Fix source names
    # with open("../data/source_typos.csv") as fin:
    #     typos = dict(map(lambda s: s.strip().split(",", 1), fin.readlines()))
    # for i, s in enumerate(index_to_source):
    #     if s in typos:
    #         index_to_source[i] = typos[s]

    embeddings = np.array([f for f in data["embedding"]], dtype=float)

    return index_to_source, source_to_index, embeddings


def read_labels(path, index_to_source=None, remove_mixed=True, remove_pro_science=True,
                drop_media_type=None):
    """
    Reads the labels CSV and filter rows to match sources in `index_to_source`.
    """
    labels = pd.read_csv(path)
    if index_to_source is not None:
        # Create new labels DataFrame in the same order as `index_to_source`
        idx_to_source_ = pd.DataFrame(dict(source=index_to_source))
        labels = idx_to_source_.join(labels.set_index("source"), on="source", how="left", lsuffix="__l")

        print("-- Could not find these labels", index_to_source[~np.isin(index_to_source, labels["source"])])

    if "conspiracy-pseudoscience" in labels.columns:
        labels["class"] = labels["conspiracy-pseudoscience"] | labels["questionable-source"]
    else:
        labels["class"] = (labels["labels.conspiracy"] > 0) | (labels["labels.pseudoscience"] > 0) | \
                          (labels["category"] == "questionable-source") | (labels["labels.factuality"] < 3)
        labels["class"] = labels["class"].astype(int)

        # Reduce leaning information to a 'leaning' column
        bias_leanings = {
            "left": "left",
            "leftcenter": "left",
            "left-center": "left",
            "extremeleft": "left",
            "center": "center",
            "right": "right",
            "center-right": "right",
            "right-center": "right",
            "extremeright": "right"
        }

        category_leanings = {
            "left": "left",
            "leftcenter": "left",
            "center": "center",
            "right": "right"
        }

        leanings = []
        for cat, bias in zip(labels["category"], labels["labels.bias"]):
            if cat in category_leanings:
                l = category_leanings[cat]
            elif bias in bias_leanings:
                l = bias_leanings[bias]
            else:
                l = "Unknown"
            leanings.append(l)

        labels["leaning"] = pd.Series(leanings)

        # Remove mixed credibility sources
        if remove_mixed:
            labels["class"][labels["labels.factuality"] == 2] = np.nan
        if remove_pro_science:
            labels["class"][labels["category"] == "pro-science"] = np.nan

        if drop_media_type:
            labels["class"][labels["media_type"].isin(drop_media_type)] = np.nan

    return labels


def read_source_details(path, index_to_source):
    """
    Reads the source details JSON file.
    """
    details = pd.read_json(path)
    details = details[details["name"].isin(index_to_source)]
    return details


def train_test_split(x, y, p=0.25, shuffle=True):
    """
    Returns a train/test split of the input data x, y.
    Args:
        x: The feature matrix
        y: The true label vector
        p: Train to test ratio (default: 0.25, 25% of the data is selected for training)
        shuffle: Shuffles dataset before splitting
    Returns:
        d_train, d_test: Split train and test datasets.
    """
    data = np.column_stack((x, y))
    if shuffle:
        np.random.shuffle(data)

    if not 0 < p < 1:
        print("Invalid value for p (%.2f), using default 0.25." % p)

    cutoff = int(len(data)*p)
    d_train = data[:cutoff]
    d_test = data[cutoff:]

    return d_train, d_test


def prepare_data(x, y, p=0.25):
    """
    Prepare data for training. Given features `x` and labels `y`, returns a train/test split of the data.
    Args:
        x: Features
        y: Labels
        p: Train/test split ratio
    """
    d_train, d_test = train_test_split(x, y, p)
    x_train = d_train[:, 0:-1]
    y_train = d_train[:, -1]
    x_test = d_test[:, 0:-1]
    y_test = d_test[:, -1]

    return x_train, x_test, y_train, y_test


def train_knn_classifier(x, y, k, metric):
    model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    model.fit(x, y)

    print(" Training accuracy:", round(accuracy_score(y, model.predict(x)), 2))

    return model


def train_label_prop_classifier(x, y, gamma=20, max_iter=1000):
    """
    Trains a Label Propagation classifier on labeled data x, y and unlabeled data x_unlabeled
    """
    model = LabelPropagation(gamma=gamma, max_iter=max_iter)
    model.fit(x, y)

    print(" Training accuracy:", round(accuracy_score(y, model.predict(x)), 2))
    return model


def evaluate(x, y, model):
    y_hat = model.predict(x)
    metrics = {
        "accuracy": round(accuracy_score(y, y_hat), 3),
        "precision": round(precision_score(y, y_hat), 3),
        "recall": round(recall_score(y, y_hat), 3),
        "f1": round(f1_score(y, y_hat), 3)
    }
    return metrics


def knn_cross_validation(x, y, k, metric, cv=5,
                         scoring=("accuracy", "precision", "recall", "f1")):
    """
    Run a cross-validation scoring using a KNN classifier on the given data.
    Args:
        x: The embedding features
        y: The true labels
        k: Number of neighbors to use in the KNN classifier
        metric: Distance metric of the classifier ("cosine" or "euclidean")
        cv: Run a `cv`-fold cross validation (default: 5).
        scoring: iterable(str) The scores to compute in the cross-validation.

    Returns:
        scores - the array of scores in each cross-validation round.

    """
    model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    scores = cross_validate(model, x, y, cv=cv, scoring=scoring, return_train_score=True)

    # Remove unwanted values
    del scores["fit_time"]
    del scores["score_time"]

    return scores


def run_fold_cross_val(embeddings, index_to_label, k, metric, cv):
    print("Folds", cv)
    print("k =", k)
    scores = knn_cross_validation(embeddings, index_to_label, k, metric, cv=cv)

    for s in scores:
        print(" - %15s\t" % s, round(scores[s].mean(), 4), "±", round(scores[s].std(), 4))

    print("===" * 20)

    return scores["test_accuracy"], scores["test_f1"]


def run_leave_one_out_cross_val(embeddings, index_to_label, k, metric, return_errors=True):
    """
    Computes a leave-one-out cross validation on the data.
    Uses n-1 data points in training and test on the 1 remaining data point.
    Args:
        embeddings: The feature vectors.
        index_to_label: The true labels (y).
        k: The number of nearest neighbors to use in the classifier.
        metric: The metric of the classifier (cosine, euclidean)
        return_errors: If True, return the set of wrongly predicted samples in each cross validation round.
    """
    loo = LeaveOneOut()
    n_splits = loo.get_n_splits(embeddings)
    acc_splits = list()

    errors = np.zeros(n_splits, dtype=int)

    for train_index, test_index in loo.split(embeddings):
        x_train, x_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = index_to_label[train_index], index_to_label[test_index]

        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(x_train, y_train)
        y_hat = model.predict(x_test)

        acc_ = accuracy_score(y_test, y_hat)
        acc_splits.append(acc_)

        errors[test_index[0]] = 1 - acc_

    acc_splits = np.array(acc_splits)

    if not return_errors:
        return acc_splits
    else:
        return acc_splits, errors


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to input embeddings")
    parser.add_argument("--known-labels", dest="known_labels", type=float, default=None,
                        help="Percentage of labels to reveal in training.")
    parser.add_argument("--metric", default=None, choices=["cosine", "euclidean"],
                        help="Metric to use in the KNN classifier.")
    parser.add_argument("--random-seed", dest="random_seed", default=42, type=int,
                        help="Random seed to use with numpy and sklearn.")
    parser.add_argument("--cv", type=int, default=20, help="No. of K-fold cross-validation.")
    parser.add_argument("--validation", type=float, default=0, help="Mount of data to allocate as validation set.")
    parser.add_argument("--leave-one-out", dest="leave_one_out",
                        action="store_true", help="Do leave-one-out cross validation.")
    parser.add_argument("--error-log", dest="error_log", type=str, default=None,
                        help="Path to save error log (as csv)")

    args = parser.parse_args()

    if args.embeddings is None:
        path_embeddings = "../model/copy+shift+stance.emb"
    else:
        path_embeddings = args.embeddings

    if args.metric is None:
        metric = 'cosine'
    else:
        metric = args.metric

    if args.known_labels is None:
        known_labels_rate = 0.2  # How many labels are known
    else:
        known_labels_rate = args.known_labels

    path_labels = "../data/source_labels.csv"
    # path_labels = "../data/labels_all.csv"

    # Load a dump of the embeddings
    index_to_source, source_to_index, embeddings = read_source_embeddings(path_embeddings)
    label_data = read_labels(path_labels, index_to_source)

    valid_labels = ~label_data["class"].isna()
    label_data = label_data[valid_labels]
    index_to_source = index_to_source[valid_labels]
    embeddings = embeddings[valid_labels]
    index_to_label = label_data["class"].astype(float).array

    np.random.seed(args.random_seed)

    # Some metadata about the experiment
    model_name = os.path.basename(path_embeddings)
    num_sources = len(index_to_source)

    print("----"*10)
    print("Model:", model_name)
    print("Sources:", num_sources)
    print("Labels: 0 - %d (%.1f%%) | 1 - %d (%.1f%%)"
          % (sum(index_to_label == 0), 100*sum(index_to_label == 0)/len(index_to_label),
             sum(index_to_label == 1), 100*sum(index_to_label == 1)/len(index_to_label)))
    print("----"*10)

    # Define some values beforehand
    k_list = np.array(range(1, 21), dtype=int)  # Try a few values of K

    print("=== Cross validation")
    k_val_data = dict()
    k_val_data["accuracy"] = list()
    k_val_data["f1"] = list()
    k_val_data["k"] = list()

    if not args.leave_one_out:
        k_acc_mean = list()
        k_f1_mean = list()
        for k in k_list:
            k_acc, k_f1 = run_fold_cross_val(embeddings, index_to_label, k, metric, args.cv)
            k_acc_mean.append(k_acc.mean())
            k_f1_mean.append(k_f1.mean())

            for acc_, f1_ in zip(k_acc, k_f1):
                k_val_data["accuracy"].append(f1_)
                k_val_data["f1"].append(acc_)
                k_val_data["k"].append(k)

        sns.relplot(data=k_val_data, x="k", y="f1", kind="line")
        plt.show()

    else:
        k_acc_mean = list()
        k_acc_std = list()
        for k in k_list:
            splits_acc, errors = run_leave_one_out_cross_val(embeddings, index_to_label, k, metric)

            print("k =", k, "mean accuracy", splits_acc.mean())

            k_acc_mean.append(splits_acc.mean())
            k_acc_std.append(splits_acc.std())

            if args.error_log:
                current_error = pd.DataFrame({"source": index_to_source, "errors-%s" % model_name: errors})
                if os.path.exists(args.error_log):
                    df_errors = pd.read_csv(args.error_log)
                    df_errors = df_errors.set_index("source").join(current_error.set_index("source"),
                                                                   how="outer", rsuffix="r")
                    gt_data = label_data[["source", "bias", "conspiracy-pseudoscience", "class"]]
                    df_errors = df_errors.join(gt_data.set_index("source"), how="outer", rsuffix="r")
                else:
                    df_errors = current_error
                df_errors.to_csv(args.error_log)

        plt.plot(k_list, k_acc_mean)
        plt.show()


if __name__ == "__main__":
    main()
