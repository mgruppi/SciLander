"""
Print comma-separated lists of sources by grouping them according to the aggregated label in nela_labels.csv
"""

from collections import defaultdict

file = "../data/nela_labels.csv"

with open(file) as fin:
    fin.readline()
    groups = defaultdict(list)
    for line in fin:
        l = line.strip().split(",")
        # l[1] contains group number, l[0] contains source name
        if l[1] != "":
            groups[l[1]].append(l[0])

    for key in sorted(groups.keys()):
        print("----")
        print(key)
        print(" ".join(groups[key]))
