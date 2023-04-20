from knn_classification import read_source_embeddings, read_labels, read_source_details, \
                                run_leave_one_out_cross_val
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors


def get_nearest_neighbors(embeddings, query_indices, k: int = 5, metric="cosine"):
    """
    Return the nearest neighbor embeddings of a list of query indices.
    Args:
        embeddings: The input embeddings
        query_indices: The indices for which to return the nearest neighbors (n elements).
        k: The number of neighbors to return
        metric: The distance metric to use
    Returns:
        neighbors: n x k matrix of nearest neighbors.
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(embeddings)
    query = [embeddings[i] for i in query_indices]
    distances, indices = nbrs.kneighbors(query)

    return indices


def run_experiment_on_embeddings(path_embeddings, path_labels,
                                 k=3,
                                 metric="cosine"):
    """
    Runs an experiment on embeddings from a given path and returns a Pandas DataFrame containing the
    errors made by the embeddings model with a KNN classifier.
    """
    # Load a dump of the embeddings
    index_to_source, source_to_index, embeddings = read_source_embeddings(path_embeddings)
    label_data = read_labels(path_labels, index_to_source)

    valid_labels = ~label_data["class"].isna()
    label_data = label_data[valid_labels]
    index_to_source = index_to_source[valid_labels]
    embeddings = embeddings[valid_labels]

    # index_to_label = np.array([label_data[s] for s in index_to_source], dtype=float)  # Map index to label (float)
    index_to_label = label_data["class"].astype(float).array

    # Some metadata about the experiment
    model_name = os.path.basename(path_embeddings).split(".")[0]
    num_sources = len(index_to_source)
    print("----"*10)
    print("Model:", model_name)
    print("Sources:", num_sources)
    print("Labels: 0 - %d (%.1f%%) | 1 - %d (%.1f%%)"
          % (sum(index_to_label == 0), 100*sum(index_to_label == 0)/len(index_to_label),
             sum(index_to_label == 1), 100*sum(index_to_label == 1)/len(index_to_label)))
    print("----"*10)

    splits_acc, errors = run_leave_one_out_cross_val(embeddings, index_to_label, k, metric)
    current_error = pd.DataFrame({"source": index_to_source, "%s" % model_name: errors})

    return current_error


def main():
    np.random.seed(42)
    path_output = "../results/errors.csv"
    # path_labels = "../data/labels_all.csv"
    path_labels = "../data/source_labels.csv"
    labels = read_labels(path_labels)

    gt_data = labels[["source", "class", "labels.bias", "labels.factuality"]]

    k = 7
    metric = "cosine"

    models_path = "../model"
    input_models = os.listdir(models_path)
    model_names = [m.split(".")[0] for m in input_models]
    input_files = [os.path.join(models_path, f) for f in input_models]

    embeddings_list = list()
    index_to_source_list = list()
    for f in input_files:
        index_to_source, source_to_index, embeddings = read_source_embeddings(f)
        embeddings_list.append(embeddings)
        index_to_source_list.append(index_to_source)

    df_errors = list()
    for path_e in input_files:
        errors = run_experiment_on_embeddings(path_e, path_labels, k, metric)
        df_errors.append(errors)

    errors = None
    for df_ in df_errors:
        if errors is None:
            errors = df_
        else:
            errors = errors.join(df_.set_index("source"), on="source", how="outer", rsuffix="r")

    # Join with ground-truth data
    errors = errors.join(gt_data.set_index("source"), on="source", how="left", rsuffix="r")

    errors.to_csv(path_output, index=None)

    for model_index, m in enumerate(model_names):
        print("==="*10)
        print("Model", m)
        errors_m = errors[~errors[m].isna()]
        print("  - Total predictions:", len(errors_m))
        print("  - Correct predictions:", sum(errors_m[m] == 0), "(%.2f)" % (sum(errors_m[m] == 0)/len(errors_m)))
        print("  - Errors:", sum(errors_m[m] == 1))
        n_errors_m = sum(errors[m] == 1)
        n_errors_0 = sum((errors_m[m] == 1) & (errors_m["class"] == 0))
        n_errors_1 = sum((errors_m[m] == 1) & (errors_m["class"] == 1))
        print("    = Class")
        print("        - Reliable (class 0):", n_errors_0, "(%.2f)" % (n_errors_0/n_errors_m))
        print("        - Unreliable (class 1):",  n_errors_1, "(%.2f)" % (n_errors_1/n_errors_m))
        print("    = Bias")
        d_bias = errors_m[errors_m[m] == 1]["labels.bias"].value_counts()
        for idx, item in zip(d_bias.index, d_bias):
            print("        - ", idx, item)


if __name__ == "__main__":
    main()
