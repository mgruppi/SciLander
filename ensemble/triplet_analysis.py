import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def fix_typos(series, typos):
    fixed = pd.Series([typos[s] if s in typos else s for s in series])
    return fixed


if __name__ == "__main__":
    path_triplets = "../data/triplets/semantic.csv"
    path_triplets = "../data/triplets_v2/shift.csv"
    path_labels = "../data/source_labels.csv"
    path_typos = "../data/source_typos.csv"

    with open(path_typos) as fin:
        typos = dict(map(lambda s: s.strip().split(","), fin.readlines()))

    triplets = pd.read_csv(path_triplets)
    columns = ["a", "p", "n"]

    labels = pd.read_csv(path_labels)
    labels["class"] = (labels["labels.factuality"] < 2) | (labels["labels.conspiracy"] > 0) \
                                                        | (labels["labels.pseudoscience"] > 0)
    labels["class"] = labels["class"].astype(int)

    labels_dict = labels[["source", "class"]].set_index("source").to_dict(orient="dict")["class"]

    for c in columns:
        triplets[c] = fix_typos(triplets[c], typos)
    triplets = triplets[(triplets["a"].isin(labels_dict))]
    triplets = triplets[(triplets["p"].isin(labels_dict))]
    triplets = triplets[(triplets["n"].isin(labels_dict))]

    grp = triplets.groupby(by=["a", "p"])
    print(grp.size())
    print(grp.nunique())

    i_a = np.array([labels_dict[s] if s in labels_dict else np.nan for s in triplets["a"]], dtype=int)
    i_p = np.array([labels_dict[s] if s in labels_dict else np.nan for s in triplets["p"]], dtype=int)
    i_n = np.array([labels_dict[s] if s in labels_dict else np.nan for s in triplets["n"]], dtype=int)

    x = np.column_stack((i_a, i_p, i_n))

    acc_pos = sum(x[:, 0] == x[:, 1])/len(x[:, 0])
    acc_neg = sum(x[:, 0] == x[:, 2])/len(x[:, 0])

    sources_included = {*set(triplets["a"]), *set(triplets["p"]), *set(triplets["n"])}
    print("-- Sources included:", len(sources_included))
    print("-- No. of triplets:", len(triplets))
    print("Accuracy positive:", acc_pos)
    print("Accuracy negative:", acc_neg)


