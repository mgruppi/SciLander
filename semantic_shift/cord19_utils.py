import csv
import json
import re
import os
import numpy as np


def extract_body_text(data: dict, max_length: int = 1e6):
    """
    Given a JSON from CORD-19, extract its text into plain text format
    Args:
        data: JSON object of the input article.
        max_length: (int) Maximum length of the output `document`. If `document`' length is greater than `max_length`,
                    it will be trimmed down to the first `max_length` characters.
    """
    if "body_text" in data:
        document = "\n".join(d["text"] for d in data["body_text"])
    if len(document) > max_length:  # Trim document to keep within a length limit
        document = document[:int(max_length)]
    return document


def read_metadata(path: str):
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


def count_publication_type(metadata: dict):
    """
    Counts and returns the lists of publications in archival and non-archival
    sources (journal/magazines vs. biorxiv, medrxiv, arxiv)
    Arguments:
        metadata - dictionary of metadata with article_id -> metadata
    Returns:
        a, b - a as the id list of archival publications (journal/magazine)
                b as the id list of non-archival papers
    """
    re_na = re.compile("biorxiv|medrxiv|arxiv")

    a = list()  # archival papers
    b = list()  # non-archival papers

    for uid in metadata:
        search = re_na.search(metadata[uid]["journal"].lower())
        if search or metadata[uid]["journal"] == "":
            b.append(uid)
        else:
            a.append(uid)

    return a, b


def read_articles_from_path(path: str, n_samples: int = None):
    """
    Generator function that yields the contents of each JSON file in `path`.
    Args:
        path: Path to `pmc_json`.
        n_samples: (int) Number of files do sample. If `None`, includes all files, otherwise draw `n_sample` files.
    Returns: data - JSON object of each file in path `data` (or a sample of).
    """
    for root, dirs, files in os.walk(path):
        if n_samples:
            file_pool = np.random.choice(files, n_samples)
        else:
            file_pool = files
        for f in file_pool:
            with open(os.path.join(root, f)) as fin:
                data = json.load(fin)
                yield data


def main():
    np.random.seed(0)
    for d in read_articles_from_path("../../data/cord19/pmc_json"):
        print(d.keys())
        print(d["paper_id"])
        print(extract_body_text(d))


if __name__ == "__main__":
    main()
