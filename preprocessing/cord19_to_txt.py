"""
Process semantic scholar articles (JSON)
Convert to plain text
"""
import json
import argparse
import os
import csv
import re
from collections import defaultdict
from preprocessing.preprocess_text import preprocess_text


def extract_plain_text(data):
    """
    Given a JSON from CORD-19, extract its text into plain text format
    """
    if "body_text" in data:
        text = "\n".join(d["text"] for d in data["body_text"])
        sentences = preprocess_text(text, use_phrases=False)
        text = ""
        for sent in sentences:
            text += " ".join(sent)
            text += "\n"
    return text


def read_metadata(path):
    """
    Reads metadata into a dictionary of article_id -> metadata
    Arguments:
        path - path to cord-19 metadata file
    Returns
        data - dictionary mapping article_id -> article metadata
    """
    with open(path) as fin:
        reader = csv.DictReader(fin, delimiter=",", quotechar="\"")
        data = dict()
        for row in reader:
            # Make pmcid column the article identifier (cord_uid)
            data[row["pmcid"]] = row
    return data


def count_publication_type(metadata):
    """
    Counts and returns the lists of publications in archival and non-archival
    sources (journal/magazines vs. biorxiv, medrxiv, arxiv)
    Arguments:
        metadata - dictionary of metadata with article_id -> metadata
    Returns:
        a, b - a as the id list of archival publications (journal/magazine)
                b as the id list of non-archival papers
    """
    re_na = re.compile("biorxiv|medrxiv")

    a = list()  # archival papers
    b = list()  # non-archival papers

    for uid in metadata:
        search = re_na.search(metadata[uid]["journal"].lower())
        if search or metadata[uid]["journal"] == "":
            b.append(uid)
        else:
            a.append(uid)

    return a, b


def main():
    """
    Crawl input directory converting JSON to plain text format
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input dir")
    parser.add_argument("metadata", type=str, help="Metadata file")
    parser.add_argument("output", type=str, help="Output file")

    args = parser.parse_args()
    metadata = read_metadata(args.metadata)

    a, b = count_publication_type(metadata)

    journals = defaultdict(int)

    # Read metadata
    for uid in a:
        journals[metadata[uid]["journal"].strip().lower()] += 1

    for k in journals:
        print(k, journals[k])

    print(len(journals))
    print(len(a), len(b))

    # read PMC json
    fout = open(args.output, "w")
    for root, dirs, files in os.walk(args.input):
        for f in files:
            f_path = os.path.join(root, f)
            with open(f_path) as fin:
                data = json.load(fin)

            # Filter for papers in year 2020
            # Skip if not found
            if data["paper_id"] not in metadata:
                continue
            if metadata[data["paper_id"]]["publish_time"] < "2020-01-01":
                continue

            text = extract_plain_text(data)
            fout.write(text)
            fout.write("\n")
    fout.close()


if __name__ == "__main__":
    main()
