from knn_classification import run_fold_cross_val, run_leave_one_out_cross_val, read_labels, read_source_embeddings
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse


def main():
    path_labels = "../data/labels_all.csv"
    path_models = "../model/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=10, help="Number of cross validation folds.")
    parser.add_argument("--k-range", dest="k_range", type=int, default=53, help="Range for KNN classifier.")
    parser.add_argument("--rounds", type=int, default=20, help="No. of rounds to evaluate (for smoother plots).")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="KNN metric.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed initializer.")

    args = parser.parse_args()
    metric = args.metric
    cv = args.cv  # Cross validation folds
    k_range = args.k_range
    rounds = args.rounds  # No of rounds (for smoother plots)
    random_seed = args.seed

    model_files = os.listdir(path_models)
    exclude_models = {"stance.emb", "jargon.emb", "copy+shift.emb", "copy+shift+stance+jargon.emb", "node2vec.emb"}
    input_embeddings = [os.path.join(path_models, f) for f in model_files if f not in exclude_models]

    model_names = [os.path.basename(f).split(".")[0] for f in input_embeddings]
    print("Models:", *model_names)

    np.random.seed(random_seed)

    k_list = list(range(1, k_range, 2))

    data = dict()
    data["model"] = list()
    data["k"] = list()
    data["accuracy"] = list()
    data["F1"] = list()

    for r in range(rounds):
        for i, path_in in enumerate(input_embeddings):
            index_to_source, source_to_index, embeddings = read_source_embeddings(path_in)
            label_data = read_labels(path_labels, index_to_source)
            valid_labels = ~label_data["class"].isna()
            label_data = label_data[valid_labels]
            embeddings = embeddings[valid_labels]
            index_to_label = label_data["class"].astype(float).array

            for k in k_list:
                k_acc, k_f1 = run_fold_cross_val(embeddings, index_to_label, k, metric, cv)
                for acc_, f1_ in zip(k_acc, k_f1):
                    data["model"].append(model_names[i])
                    data["k"].append(k)
                    data["accuracy"].append(acc_)
                    data["F1"].append(f1_)


    sns.set(context='paper', style='white', color_codes=True, font_scale=1.5)
    sns.color_palette('colorblind')
    hue_order=['SciLander', 'SciLander (shift)', 'SciLander (copy)', 'Stylistic', 'BERT', 'SciBERT', 'BERT+node2vec', 'SciBERT+node2vec']
    dashes=[(2,1),(1,0),(2,1),(1,0),(2,1),(2,1),(1,0),(2,1)]
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    g = sns.relplot(data=data, x="k", y="F1", kind="line", style="model", hue="model", palette='colorblind', linewidth = 3, ci=None, hue_order=hue_order, dashes=dashes)
    # plt.grid(True)
    g.legend.remove()
    g.add_legend()
    leg = plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.4), loc="upper center")
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    # plt.title("Model Comparison")
    plt.savefig("../results/figures/model_comparison.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
