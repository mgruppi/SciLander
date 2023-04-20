import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from knn_classification import read_labels, read_source_embeddings

path_results = "../results/"
path_figures = os.path.join(path_results, "figures")


color_palette = [
                "#003f5cEE",
                "#2f4b7cEE",
                "#665191EE",
                "#a05195EE",
                "#d45087EE",
                "#f95d6aEE",
                "#ff7c43EE",
                "#ffa600EE"
]


def pca_plot(embeddings, label=None, palette="colorblind",
             class_name="Class",
             hue="Class",
             cluster_centers=None,
             cluster_labels=None,
             show_legend=True,
             show_colorbar=False,
             title=None,
             cbar_labels=None,
             cbar_tick_locations=None,
             show_grid=False,
             show_title=False,
             kind="scatterplot",
             pca=True,
             **kwargs):

    plt.tight_layout()
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)

    if pca:
        decomp = PCA(n_components=2, whiten=True)
    else:
        decomp = TSNE(n_components=2, metric="cosine")

    x = decomp.fit_transform(embeddings)
    data = {"x": x[:, 0], "y": x[:, 1], class_name: label,
            "label": cluster_labels}

    if kind == "scatterplot":
        axes = sns.scatterplot(data=data, x="x", y="y", palette=palette, hue=hue, **kwargs)
    elif kind == "kdeplot":
        axes = sns.kdeplot(data=data, x="x", y="y", palette=palette, hue=hue, **kwargs)
    elif kind == "kde+scatter":
        axes = sns.kdeplot(data=data, x="x", y="y", palette=palette, hue=hue, **kwargs)
        sns.scatterplot(data=data, x="x", y="y", color="black")
    else:
        print("* ERROR: Invalid 'kind' argument for pca_plot:", kind)
        return

    if cluster_centers is not None:
        x_centroids = decomp.transform(cluster_centers)
        plt.scatter(x_centroids[:, 0], x_centroids[:, 1], color="black")

    if show_colorbar:
        norm = plt.Normalize(min(label), max(label))
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        if cbar_labels:
            print("setting cbar")
            cbar.ax.get_yaxis().set_ticks([])
            cbar.ax.get_yaxis().set_ticks(cbar_tick_locations)
            cbar.ax.get_yaxis().set_ticklabels(cbar_labels)
    # else:
    #     # We will sort the legend by cluster name
    #     axes.legend().remove()
    #     axes.add_legend()
    #     leg = axes.legend(ncol=2)

    plt.grid(show_grid)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if not show_legend:
        plt.legend().remove()
    else:
        patch = [mpatches.Patch(color=c, label=l) for c, l in zip(sns.color_palette("Set2", 7), ['A', 'B', 'C', 'D', 'E', 'F', 'G'])]
        plt.legend(handles=patch, ncol=1, bbox_to_anchor=(1.3, 1), loc="upper right", title='Cluster')

    if title:
        if show_title:
            plt.title(title)
            
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.savefig(os.path.join(path_figures, title+".pdf"), format="pdf", bbox_inches='tight')
    else:
        plt.show()

    plt.clf()


def plot_multidimensional_scaling(embeddings, colors=None, metric="euclidean"):

    if metric != "cosine":
        mds = MDS(n_components=2, dissimilarity=metric)
        x = mds.fit_transform(embeddings)
    else:
        d = pairwise_distances(embeddings, metric="cosine")
        mds = MDS(n_components=2, dissimilarity="precomputed")
        x = mds.fit_transform(d)

    plt.scatter(x[:, 0], x[:, 1], color=colors)
    plt.show()

    return x


def plot_spectral_embedding(embeddings, colors=None):
    spectral = SpectralEmbedding(n_components=2)

    x = spectral.fit_transform(embeddings)

    plt.scatter(x[:, 0], x[:, 1], color=colors)
    plt.show()
    return x


def analyze_clusters(embeddings, index_to_cluster, index_to_label, index_to_leaning, index_to_conspiracy,
                     index_to_pseudoscience, index_to_factuality, index_to_core=None, model_name=""):
    cluster_indices = np.unique(index_to_cluster)
    unreliability_scores = np.zeros(len(cluster_indices), dtype=float)
    leaning_scores = np.zeros(len(cluster_indices), dtype=float)
    purity_scores = np.zeros(len(cluster_indices), dtype=float)
    conspiracy_scores = np.zeros(len(cluster_indices), dtype=float)
    partisanship_scores = np.zeros(len(cluster_indices), dtype=float)

    non_noisy_mask = (index_to_cluster != -1)  # Check if any cluster is labeled as -1
    embeddings = embeddings[non_noisy_mask]
    index_to_cluster = index_to_cluster[non_noisy_mask]
    index_to_label = index_to_label[non_noisy_mask]
    index_to_leaning = index_to_leaning[non_noisy_mask]
    index_to_conspiracy = index_to_conspiracy[non_noisy_mask]
    index_to_pseudoscience = index_to_pseudoscience[non_noisy_mask]
    index_to_factuality = index_to_factuality[non_noisy_mask]
    if index_to_core is not None:
        index_to_core = index_to_core[non_noisy_mask]
    else:
        index_to_core = np.ones(len(index_to_cluster), dtype=bool)

    cluster_scores = dict()

    for i in cluster_indices:
        if i == -1:
            continue
        cluster_scores[i] = dict()

        cluster_mask = index_to_cluster == i

        idx_labels = index_to_label[cluster_mask]  # Indices of the points within current cluster

        unreliable_count = sum(idx_labels == 1)
        reliable_count = sum(idx_labels == 0)
        print(" => cluster", chr(65+i), sum(cluster_mask))

        # Percentage of unreliable to total number of sources
        unreliability_scores[i] = unreliable_count/(unreliable_count + reliable_count)
        print("   + Class:", unreliability_scores[i], "- reliable:", reliable_count, "| unreliable:", unreliable_count)
        # Purity score = n_majority_class / cluster_size
        purity_scores[i] = max(unreliable_count, reliable_count)/(reliable_count+unreliable_count)
        # print("   + Purity:", purity_scores[i])
        # Aggregate political leaning scores in the cluster
        leanings = index_to_leaning[cluster_mask & index_to_core]
        leanings = np.nan_to_num(leanings, 0)
        leaning_scores[i] = leanings.sum() / len(leanings)
        partisanship_scores[i] = np.abs(leaning_scores[i])

        # leanings, counts = np.unique(leanings, return_counts=True)
        # max_leaning = leanings[np.argmax(counts)]
        # leaning_scores[i] = max_leaning

        # print("    + Leaning", leaning_scores[i])
        # print("      - Breakdown", np.unique(leanings, return_counts=True))

        conspiracy_values = index_to_conspiracy[cluster_mask]
        pseudoscience_values = index_to_pseudoscience[cluster_mask]

        consp_density = sum((conspiracy_values > 0) | (pseudoscience_values > 0))/len(conspiracy_values)
        conspiracy_scores[i] = consp_density
        # print("    + Conspiracy", np.unique(conspiracy_values, return_counts=True), "DENSITY", consp_density)
        # print("    + Pseudoscience", np.unique(pseudoscience_values, return_counts=True))

        fact_values = index_to_factuality[cluster_mask]
        # print("    + Factuality", np.unique(fact_values, return_counts=True))

        cluster_scores[i]["unreliable_density"] = unreliability_scores[i]
        cluster_scores[i]["purity"] = purity_scores[i]
        cluster_scores[i]["leaning"] = leaning_scores[i]
        cluster_scores[i]["conspiracy_density"] = conspiracy_scores[i]
        cluster_scores[i]["partisanship"] = partisanship_scores[i]

    # non_noisy_mask = (index_to_cluster != -1)  # Check if any cluster is labeled as -1
    # embeddings = embeddings[non_noisy_mask]
    # index_to_cluster = index_to_cluster[non_noisy_mask]
    # # index_to_label = index_to_label[non_noisy_mask]
    # # index_to_leaning = index_to_leaning[non_noisy_mask]
    # # index_to_conspiracy = index_to_conspiracy[non_noisy_mask]
    # # index_to_pseudoscience = index_to_pseudoscience[non_noisy_mask]
    # # index_to_factuality = index_to_factuality[non_noisy_mask]

    cluster_labels = ["%s" % chr(i + ord('A')) for i in index_to_cluster]
    cluster_order = sorted(np.unique(cluster_labels))
    pca_plot(embeddings, label=cluster_labels, hue="Cluster", style="Cluster", class_name="Cluster",
             # palette=sns.color_palette("husl", len(np.unique(cluster_labels))),
             palette=sns.color_palette("Set2", len(np.unique(cluster_labels))),
             hue_order=cluster_order,
             # cluster_centers=clusters.cluster_centers_,
             show_colorbar=False,
             title="clusters_%s" % model_name,
             kind="kdeplot", fill=True, thresh=0.05)

    labels = [round(unreliability_scores[i], 2) for i in index_to_cluster]
    labels = [l[0] for l in MinMaxScaler().fit_transform(np.array(labels).reshape(-1, 1))]
    pca_plot(embeddings, label=labels, hue="Unreliable_Ratio",
             class_name="Unreliable_Ratio",
             cluster_labels=cluster_labels,
             # style="label",
             palette=sns.diverging_palette(145, 300, s=60, as_cmap=True),
             show_legend=False,
             show_colorbar=True,
             title="clusters_unreliable_density_%s" % model_name,
             kind="kdeplot", fill=True, thresh=0.05)

    labels = [round(leaning_scores[i], 2) for i in index_to_cluster]
    pca_plot(embeddings, label=labels, class_name="Leaning", hue="Leaning",
             palette=sns.diverging_palette(240, 10, center="light", as_cmap=True),
             cluster_labels=cluster_labels,
             # style="label",
             show_legend=False,
             show_colorbar=True,
             title="clusters_political_leaning_%s" % model_name,
             cbar_labels=["Left", "Unknown", "Right"],
             cbar_tick_locations=[min(labels), (min(labels)+max(labels))/2, max(labels)],
             kind="kdeplot", fill=True, thresh=0.05)

    labels = [round(conspiracy_scores[i], 2) for i in index_to_cluster]
    pca_plot(embeddings, label=labels, class_name="Conspiracy", hue="Conspiracy",
             palette=sns.diverging_palette(120, 280, center="light", as_cmap=True),
             # palette=sns.color_palette("rocket", as_cmap=True),
             cluster_labels=cluster_labels,
             # style="label",
             show_legend=False,
             show_colorbar=True,
             title="clusters_conspiracy_density_%s" % model_name,
             cbar_labels=["No Conspiracy", "", "High Conspiracy"],
             cbar_tick_locations=[min(labels), (min(labels)+max(labels))/2, max(labels)],
             kind="kdeplot", fill=True, thresh=0.05)

    labels = [round(partisanship_scores[i], 2) for i in index_to_cluster]
    pca_plot(embeddings, label=labels, class_name="Partisanship", hue="Partisanship",
             palette=sns.light_palette("orange", as_cmap=True),
             # palette=sns.color_palette("rocket", as_cmap=True),
             cluster_labels=cluster_labels,
             # style="label",
             show_legend=False,
             show_colorbar=True,
             title="clusters_partisanship_%s" % model_name,
             cbar_labels=["Weak", "", "Strong"],
             cbar_tick_locations=[min(labels), (min(labels)+max(labels))/2, max(labels)],
             kind="kdeplot", fill=True, thresh=0.05)

    return cluster_scores


def get_cluster_cores(embeddings, clusters, index_to_source):
    cluster_indices = np.unique(clusters.labels_)
    cores = dict()
    nbrs = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(embeddings)
    for i in cluster_indices:
        k_dists, k_cores = nbrs.kneighbors([clusters.cluster_centers_[i]])
        cores[i] = dict()
        cores[i]["cores"] = index_to_source[k_cores][0]
        cores[i]["distances"] = k_dists[0]
    return cores


def main():
    path_embeddings = "../model/SciLander.emb"
    path_labels = "../data/source_labels.csv"

    path_output_clusters = "../results/source_clusters.csv"
    path_output_cores = "../results/cluster_cores.csv"

    if not os.path.exists(path_results):
        os.makedirs(path_results)
    if not os.path.exists(path_figures):
        os.makedirs(path_figures)

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default=None, help="Path to embeddings")
    parser.add_argument("--k", type=int, default=10, help="No. of clusters for K-Means")
    parser.add_argument("--eps", type=float, default=0.1, help="DBSCAN eps.")
    parser.add_argument("--min-samples", dest="min_samples", type=int, default=5, help="DBSCAN min_samples arg.")
    parser.add_argument("--use-dbscan", dest="use_dbscan", action="store_true", help="Use DBSCAN clustering")

    args = parser.parse_args()

    if args.embeddings:
        path_embeddings = args.embeddings

    model_name = os.path.basename(path_embeddings).split(".")[0]

    np.random.seed(42)

    index_to_source, source_to_index, embeddings = read_source_embeddings(path_embeddings)
    label_data = read_labels(path_labels, index_to_source, drop_media_type=["Organization/Foundation"])

    valid_labels = ~label_data["class"].isna()
    label_data = label_data[valid_labels]
    index_to_source = index_to_source[valid_labels]
    index_to_name = np.array(label_data["display_name"].str.replace(r" - Media Bias/Fact Check", ""))
    embeddings = embeddings[valid_labels]

    index_to_label = label_data["class"].astype(int)
    index_to_conspiracy = np.nan_to_num(label_data["labels.conspiracy"], -1)
    index_to_pseudoscience = np.nan_to_num(label_data["labels.pseudoscience"], -1)
    index_to_factuality = np.nan_to_num(label_data["labels.factuality"], -1)
    index_to_leaning = np.nan_to_num(label_data["leaning"], -1)
    index_to_is_conspiracy = ((index_to_conspiracy > 0) | (index_to_pseudoscience > 0)).astype(int)

    leaning_labels = {"left": "Left", "right": "Right", "center": "Center", "Unknown": "Unknown"}
    index_to_class = ["Reliable" if y == 0 else "Unreliable" for y in index_to_label]
    index_to_leaning_label = [leaning_labels[i] for i in index_to_leaning]
    pca_plot(embeddings, label=index_to_class, title="reliable_unreliable_%s" % model_name,
             kind="kdeplot", fill=True, thresh=0.2)

    pca_plot(embeddings, label=index_to_leaning_label, title="political_leaning_%s" % model_name,
             kind="kdeplot", fill=True, thresh=0.2)

    pca_plot(embeddings, label=index_to_is_conspiracy, title="is_conspiracy_%s" % model_name,
             kind="kdeplot", fill=True, thresh=0.2)

    pca_plot(embeddings, label=label_data["category"], title="category_%s" % model_name,
             kind="kdeplot", fill=True, thresh=0.2)

    pca = PCA(n_components=3)
    components = pca.fit_transform(embeddings)
    pc_1 = components[:, 0]
    pc_2 = components[:, 1]
    pc_3 = components[:, 2]

    leaning_mask = (np.isin(index_to_leaning_label, ["Left", "Right"]))
    pca_leanings = index_to_leaning[leaning_mask]
    pc_2 = pc_2[leaning_mask]

    data = {"PC1": pc_1, "PC2": pc_2, "PC3": pc_3,
            "Credibility": index_to_class, "Pol. Leaning": pca_leanings,
            "is_conspiracy": index_to_is_conspiracy}

    sns.kdeplot(data=data, x="PC1", hue="Credibility", fill=True, palette="PRGn", hue_order=["Unreliable", "Reliable"])
    plt.tight_layout()
    plt.savefig("../results/figures/pca_class.pdf", format="pdf")
    plt.clf()

    sns.kdeplot(data=data, x="PC2", hue="Pol. Leaning", fill=True, palette="coolwarm")
    plt.tight_layout()
    plt.savefig("../results/figures/pca_leaning.pdf", format="pdf")
    plt.clf()

    print("Finding clusters...")
    eps = args.eps
    n_clusters = args.k
    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)
    dbscan = DBSCAN(eps=eps, metric="cosine", min_samples=args.min_samples).fit(embeddings)

    index_to_kmeans_cluster = kmeans.labels_
    index_to_dbscan_cluster = dbscan.labels_

    if not args.use_dbscan:
        index_to_cluster = index_to_kmeans_cluster
    else:
        index_to_cluster = index_to_dbscan_cluster

    # Save source_clusters file
    df_clusters = pd.DataFrame({"source": index_to_source, "cluster": index_to_cluster})
    df_clusters.to_csv(path_output_clusters, index=None)

    bias, counts = np.unique(index_to_label, return_counts=True)
    print(*zip(bias, counts))
    print("------"*10)

    leaning_weights = {
        "extremeleft": -1,
        "left": -.5,
        "leftcenter": -.25,
        "center": 0,
        "rightcenter": .25,
        "right": .5,
        "extremeright": 1
    }

    # leaning_weights = {
    #     "extremeleft": -1,
    #     "left": -1,
    #     "leftcenter": -1,
    #     "center": 0,
    #     "rightcenter": 1,
    #     "right": 1,
    #     "extremeright": 1
    # }

    print("-- CLUSTERS", np.unique(index_to_cluster, return_counts=True))

    # Get core samples for each cluster
    core_samples = dict()
    index_to_core = np.zeros(len(index_to_cluster), dtype=bool)
    index_to_core[dbscan.core_sample_indices_] = True  # Create a binary mask of index-to-core values

    for idx in dbscan.core_sample_indices_:
        if index_to_cluster[idx] not in core_samples:
            core_samples[index_to_cluster[idx]] = list()
        core_samples[index_to_cluster[idx]].append(idx)
    print("-- DBSCAN core samples")
    # Write core samples CSV
    core_names = dict()
    with open(path_output_cores, "w") as fout:
        fout.write("source,cluster\n")
        for k in core_samples:
            core_samples[k] = np.array(core_samples[k])
            x_cluster = embeddings[core_samples[k]]
            nbrs = NearestNeighbors(n_neighbors=len(x_cluster), metric="cosine").fit(x_cluster)
            cluster_centroid = np.mean(x_cluster, axis=0)
            distances, indices = nbrs.kneighbors([cluster_centroid])
            print("  Cluster:", k, index_to_name[core_samples[k][indices[0]]])
            core_names[k] = index_to_name[core_samples[k][indices[0]]]
            print("--")
            for src in index_to_source[core_samples[k][indices[0]]]:
                fout.write("%s,%d\n" % (src, int(k)))

    index_to_leaning = np.fromiter((leaning_weights[b] if b in leaning_weights else np.nan
                                    for b in label_data["labels.bias"]), dtype=float)

    cores = get_cluster_cores(embeddings, kmeans, index_to_source)
    cluster_scores = analyze_clusters(embeddings, index_to_cluster, index_to_label, index_to_leaning,
                                      index_to_conspiracy, index_to_pseudoscience, index_to_factuality,
                                      index_to_core=index_to_core,
                                      model_name=model_name)

    # Print-out Cluster Tables
    for k in sorted(core_samples):
        cluster_name = "%s" % chr(65+int(k))
        d_consp = cluster_scores[k]["conspiracy_density"]
        d_unr = cluster_scores[k]["unreliable_density"]
        d_partisan = cluster_scores[k]["partisanship"]
        p_leaning = cluster_scores[k]["leaning"]
        cores = ",".join(core_names[k][:10])

        print("%s\t&\t%.2f\t&\t%.2f\t&\t%s \\\\" % (cluster_name, d_consp, d_partisan, cores))

    # leaning_thresholds = [-0.5, -0.2, 0.2, 0.5, 1]
    # leaning_labels = ["left", "left-center", "center", "right-center", "right"]
    # for k in cores:
    #     cluster_title = "Cluster %s" % chr(65+k)
    #     cluster_cores = ", ".join(cores[k]["cores"])
    #
    #     cluster_leaning = cluster_scores[k]["leaning"]
    #
    #     print("%s & %.2f & %s & %s \\\\" % (cluster_title, cluster_scores[k]["unreliable_density"],
    #                                         cluster_leaning, cluster_cores))


if __name__ == "__main__":
    main()
