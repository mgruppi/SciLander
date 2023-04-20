"""
Functions used to create a source network graph G(V,E) where node set V are the sources and edge set E
are the connections defined by one of the provided methods.
"""
import argparse
from collections import defaultdict
import networkx as nx


def jaccard_index(a, b):
    """
    Computes the Jaccard Index for a pair of input sets.
    The jacard index J(a,b) is computed as J(a,b) = |a ^ b|/|a U b| (intersection/union)
    Arguments:
        a       - set, first set
        b       - set, second set
    Return:
        j       - the jaccard index (similarity) between a and b.
    """
    union_ab = set.union(a, b)
    inter_ab = set.intersection(a, b)
    return len(inter_ab)/len(union_ab)


def main():
    parser = argparse.ArgumentParser()

    output_file = "data/graph.gml"
    labels_file = "data/nela_labels.csv"

    labels = dict()
    with open(labels_file) as fin:
        fin.readline()  # read header
        for line in fin:
            line_list = line.strip().split(",")
            if line_list[1] != "":
                labels[line_list[0]] = line_list[1]

    f_path = "data/references/sources-references_no_duplicates.csv"
    citations = defaultdict(set)  # source->citations mapping

    with open(f_path) as fin:
        fin.readline()
        for line in fin:
            source, article_url = line.strip().split(",", 1)
            source = source.split(".")[0]  # Remove top-level domain from source
            citations[source].add(article_url)

    sources = list(citations.keys())

    edge_list = list()  # stores weighted edge list (a, b, w)
    for i, s_a in enumerate(sources):
        for j, s_b in enumerate(sources[i+1:]):
            j_index = jaccard_index(citations[s_a], citations[s_b])
            if j_index > 0.0:
                edge_list.append((s_a, s_b, j_index))

    # Create graph
    print("Creating graph...")
    G = nx.Graph()
    for e in edge_list:
        G.add_edge(e[0], e[1], weight=e[2])

    # Set node attributes
    to_remove = list()
    for i, n in enumerate(G.nodes()):
        if n in labels:
            G.nodes[n]["credibility_score"] = labels[n]
        else:
            to_remove.append(n)

    for n in to_remove:
        G.remove_node(n)

    print(len(G.nodes()), "nodes |", len(G.edges()), "edges")
    nx.write_gml(G, output_file)


if __name__ == "__main__":
    main()
