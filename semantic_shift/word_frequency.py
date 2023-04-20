"""
Count word frequency of words in a given text file
Writes a dictionary file containing
<word> <count> <frequency>
Where <count> is the raw word count and <frequency> is the relative frequency, i.e., normalized by the total word count
"""
import argparse
from collections import defaultdict
import re
import os
import json
import time
import sqlite3
from multiprocessing import Pool


def merge_results(results):
    """
    Merges embeddings from a Pool into a single dictionary keyed by source name.
    :param results: Tuple of dictionaries.
    :return d: All dictionaries from `result` combined.
    """
    d = dict()

    for r in results:
        d.update(r)
    return d


def get_frequency_dict(batch):
    """
    Compute word counts and relative frequencies for words from a corpus in an input file.
    :param batch: List object [corpus, source_name] where corpus is a single string and source_name a string.
    :return d: dictionary of word counts and frequencies.
    """
    d_freq = defaultdict(float)
    d_count = defaultdict(int)

    corpus = batch[0]
    source_name = batch[1]

    regex = re.compile("\s+")
    # with open(path) as fin:
    #     for line in fin:
    #         tokens = regex.split(line)
    #         for t in tokens:
    #             d_count[t] += 1
    #             total_count += 1

    tokens = regex.split(corpus)
    for t in tokens:
        d_count[t] += 1
    total_count = len(tokens)

    for word in d_count:
        d_freq[word] = d_count[word]/total_count

    return {source_name: {"count": d_count}}


def update_frequency_dict(data, new):
    """
    Update frequency dictionary with a new batch of data `new`.
    Adds new values to existing word counts.
    """
    for src in new:
        if src in data:
            data[src]["count"] = {w: data[src]["count"].get(w, 0) +
                                  new[src]["count"].get(w, 0)
                                  for w in set(data[src]["count"]) | set(new[src]["count"])}
        else:
            data[src] = new[src]
    return data


def compute_relative_frequency(data, src):
    """
    Computes the relative frequency of words in a frequency dictionary.
    """
    num_words = sum(c for c in data[src]["count"].values())
    d_freq = {}
    for word in data[src]["count"]:
        d_freq[word] = data[src]["count"][word]/num_words
    data[src]["freq"] = d_freq
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_db", type=str, help="Path to input database.")
    parser.add_argument("output_file", type=str, help="Path to output file.")
    parser.add_argument("--workers", type=int, default=None, help="No. of jobs to run in parallel.")
    parser.add_argument("--lower", action="store_true", help="Convert all terms to lower case.")

    args = parser.parse_args()

    con = sqlite3.connect(args.input_db)
    query = "SELECT source, content, date FROM newsdata"
    t0 = time.time()
    data = dict()
    r = con.cursor().execute(query)

    job_count = 0
    while True:
        print(job_count)
        batch = r.fetchmany(100000)
        if batch is None or len(batch) <= 0:
            print()
            break
        job_count += len(batch)
        batch_jobs = [[b[1], b[0]] for b in batch]
        with Pool(args.workers) as p:
            results = p.map(get_frequency_dict, batch_jobs)
        data = update_frequency_dict(data, merge_results(results))

    print(len(data))

    sources = data.keys()
    for src in sources:
        compute_relative_frequency(data, src)

    if not os.path.exists(os.path.dirname(args.output_file)):
        try:
            os.makedirs(os.path.dirname(args.output_file))
        except Exception as e:
            print(e)

    with open(args.output_file, "w") as fout:
        json.dump(data, fout)

    print("Done in %.2f seconds." % (time.time() - t0))


def main_directory():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to input corpus directory/file.")
    parser.add_argument("output_file", type=str, help="Path to save output file.")
    parser.add_argument("--is_input_file", action="store_true", help="Input is file instead of directory.")
    parser.add_argument("--workers", type=int, default=None, help="No. of worker processes.")

    args = parser.parse_args()

    t0 = time.time()
    corpus_files = list()
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            print("File %s" % f)
            corpus_files.append(os.path.join(root, f))

    with Pool(args.workers) as p:
        results = p.map(get_frequency_dict, corpus_files)

    data = merge_results(results)

    if not os.path.exists(os.path.dirname(args.output_file)):
        try:
            os.makedirs(os.path.dirname(args.output_file))
        except Exception as e:
            print(e)

    with open(args.output_file, "w") as fout:
        json.dump(data, fout)

    print("Done in %.2f seconds." % (time.time() - t0))


if __name__ == "__main__":
    main()
