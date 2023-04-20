"""
Merges the list of URL,sources with the corresponding credibility label for that source from NELA nela_labels.csv
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", type=str, help="Path to URL+source file")
    parser.add_argument("labels", type=str, help="Path to labels file")
    parser.add_argument("output", type=str, help="Path to output file")
    args = parser.parse_args()

    with open(args.urls) as fin:
        data_url = list(map(lambda s: s.strip().split(","), fin.readlines()))
    with open(args.labels) as fin:
        fin.readline()  # read header
        labels = dict()
        for line in fin:
            l = line.strip().split(",")
            labels[l[0]] = l[1]

    sources_not_found = set()
    with open(args.output, "w") as fout:
        for d in data_url:
            if d[1] in labels:
                fout.write("%s,%s,%s\n" % (d[0], d[1], labels[d[1]]))
            else:
                sources_not_found.add(d[1])

    for s in sorted(sources_not_found):
        print(s)
    print("--", len(sources_not_found), "sources not found")


if __name__ == "__main__":
    main()