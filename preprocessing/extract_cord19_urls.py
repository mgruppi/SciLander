"""
Extract URLs from Cord-19 articles
"""

import argparse
from preprocessing.timeline_cord19 import read_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to CORD-19 meta data")
    parser.add_argument("output", type=str, help="Path to output text file")

    args = parser.parse_args()

    data = read_metadata(args.input)
    print(data[0].keys())

    with open(args.output, "w") as fout:
        fout.write("url,doi\n")
        for d in data:
            fout.write("%s,%s\n" % (d["url"], d["doi"]))
    print("Done.")


if __name__ == "__main__":
    main()