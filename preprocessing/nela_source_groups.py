import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("labels", type=str, help="Path to labels file.")

args = parser.parse_args()

with open(args.labels) as fin:
    fin.readline()  # read header
    data = list(map(lambda s: s.strip().split(",", 1), fin.readlines()))

label_names = {"0": "reliable", "1": "unreliable", "2": "mixed"}

source_labels = defaultdict(list)

for d in data:
    source_labels[d[1]].append(d[0])

for key in source_labels:
    label_name = label_names[key]
    with open("nela-sources-%s.txt" % label_name, "w") as fout:
        fout.write(" ".join(source_labels[key]))
