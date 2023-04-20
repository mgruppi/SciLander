import networkx
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import argparse


def generate_second_order_triplets(g, n_samples=100):
    """
    Generate samples of triplets in g.
    For each node `u` in g, sample direct neighbors of `u` and make them a positive
    training pair. The probability of sampling a neighbor `v` is based on the weight of the out-edges of `u`
    p(v) = e(u,v)['weight']/Z, where `Z` is the normalizing factor given by the sum of out weights of v.
    Negative pairs are obtained by randomly sampling a node in g.
    Args:
        g: NetowrkX graph
        n_samples: Number of triplets to generate (per source node)

    Returns:
        triplets (list(tuple)) - The output list of triplets generated from the random walks.
    """
    triplets = list()
    np.random.seed(0)

    for u in g.nodes:
        # Get all outgoing edges from u
        out_edges = g.out_edges(u, data=True)
        z = np.sum(np.fromiter((e[2]["weight"] for e in out_edges), dtype=float))  # normalizing factor
        neighbors = [v for v in g.neighbors(u)]

        if len(neighbors) < 2:
            continue

        p = np.array([g.edges[(u, v)]["weight"]/z for v in neighbors])

        positives = [tuple(np.random.choice(neighbors, size=2, p=p, replace=False)) for i in range(n_samples)]
        negatives = np.random.choice(g.nodes, size=n_samples)

        t = [(p[0], p[1], n) for p, n in zip(positives, negatives)]

        triplets.extend(t)
    return triplets


def main():
    path_in = "nela-covid.gml"
    path_out = "../data/triplets/content-sharing.csv"
    path_typos = "../data/source_typos.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of samples to draw, per source.")

    args = parser.parse_args()

    with open(path_typos) as fin:
        def parse_line(l):
            return l.strip().split(",", 1)
        typos = dict(map(parse_line, fin.readlines()))

    g = networkx.readwrite.read_gml(path_in)
    triplets = generate_second_order_triplets(g, n_samples=args.n)

    with open(path_out, "w") as fout:
        fout.write("a,p,n\n")
        for a, p, n in triplets:
            # Fix source names by applying the typos dictionary
            a = typos[a] if a in typos else a
            p = typos[p] if p in typos else p
            n = typos[n] if n in typos else n

            fout.write("%s,%s,%s\n" % (a, p, n))


if __name__ == "__main__":
    main()
