from triplet_embeddings import prepare_triplets, train_embeddings
import os
import pandas as pd
import argparse
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
from utils import timed


@timed
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Path to output embeddings")
    parser.add_argument("--threshold", type=int, default=0, help="Drop triplets with less than threshold occurrences.")
    parser.add_argument("--indicators", type=str, nargs="+", default=set(),
                        help="Name of the indicators to use.")
    parser.add_argument("--exclude", nargs="+", type=str, default=set(),
                        help="Exclude triplets when embedding {content-sharing, semantic, stance_triplets}."
                             "NOTE: Only used if argument --indicators is not passed.")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine",
                        help="Distance metric to use with the margin loss function.")
    parser.add_argument("--margin", type=float, default=3, help="Margin to be used in the triplet loss.")
    parser.add_argument("--use-cuda", dest="use_cuda", action="store_true",
                        help="Use CUDA for training on GPU.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Name of the CUDA GPU device to use.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs to training embeddings on.")
    parser.add_argument("--dim", type=int, default=50, help="Dimension of output embeddings.")

    args = parser.parse_args()

    if args.output:
        path_embeddings = args.output
    else:
        path_embeddings = "../model/source_embeddings.emb"

    if args.indicators:
        args.indicators = set(args.indicators)
    elif args.exclude:
        args.exclude = set(args.exclude)

    thresh = args.threshold

    root_triplets = "../data/triplets/"
    triplet_files = sorted(os.listdir(root_triplets))
    print(triplet_files)
    path_triplets = [os.path.join(root_triplets, f)
                     for f in triplet_files
                     # if f.split(".")[0] not in args.exclude
                     if f.split(".")[0] in args.indicators]
    print("Using", path_triplets)
    triplet_dfs = [pd.read_csv(f) for f in path_triplets]

    df_all = pd.concat(triplet_dfs)
    # df_all["indicator"] = "all_indicators"
    df_all["indicator"] = sum([[f]*len(triplet_dfs[i]) for i, f in enumerate(args.indicators)], [])

    triplets, source_index = prepare_triplets(triplet_dfs=[df_all], triplet_threshold=thresh)
    print("---" * 10)

    embedding_params = {
        "embedding_dim": args.dim,
        "lr": 1e-2,
        "epochs": args.epochs,
        "batch_size": 1024,
        "margin": args.margin,
        "metric": args.metric,
        "use_cuda": args.use_cuda,
        "device": args.device
    }

    print("Training source embeddings")

    embeddings = train_embeddings(triplets, source_index, **embedding_params)

    with open(path_embeddings, "wb") as fout:
        # Dump embeddings to file
        pickle.dump(embeddings, fout)

    source_to_index = {s: i for i, s in enumerate(embeddings["source"])}

    x = np.array([f for f in embeddings["embedding"]], dtype=float)
    print(x.shape)

    nbrs = NearestNeighbors(n_neighbors=10, metric=embedding_params["metric"]).fit(x)

    distances, indices = nbrs.kneighbors()

    queries = ["abcnews", "infowars", "veteranstoday", "thehill"]

    for q in queries:
        try:
            source_query = q
            query_index = source_to_index[source_query]

            print("---", q)
            print([embeddings["source"][i] for i in indices[query_index]])
        except KeyError as e:
            print("Source not found", e)


if __name__ == "__main__":
    main()

