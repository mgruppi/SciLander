"""
Given a list of NELA article URLs, extract the corresponding sources from a NELA database
"""

import argparse
import sqlite3


def get_sources_from_url(conn):
    """
    Executes the query to retrieve sources given an article URL from a NELA database
    Requires a list of URLs and a connection to a NELA sqlite3 database .
    Returns a list of source names. If a URL cannot be found, returns None instead.
    """

    sources = dict()

    query = "SELECT url,source FROM newsdata"
    r = dict(conn.cursor().execute(query).fetchall())
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=str, help="Path to NELA database")
    parser.add_argument("urls", type=str, help="Path to URL list")
    parser.add_argument("output", type=str, help="Path to output list")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)

    with open(args.urls) as fin:
        # read header
        print("Reading URLs")
        fin.readline()
        urls = list(map(lambda s: s.strip().split(",")[0], fin.readlines()))

    print("Searching sources...")
    sources = get_sources_from_url(conn)

    mappings = dict()

    with open(args.output, "w") as fout:
        for u in urls:
            if u in sources:
                mappings[u] = sources[u]
        for u in mappings:
            fout.write("%s,%s\n" % (u, mappings[u]))


if __name__ == "__main__":
    main()