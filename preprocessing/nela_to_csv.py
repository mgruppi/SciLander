import argparse
import sqlite3


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to NELA database")
parser.add_argument("output", type=str, help="Path to output CSV")

args = parser.parse_args()

con = sqlite3.connect(args.path)

results = con.execute("SELECT source, url, date(published_utc, 'unixepoch') FROM newsdata").fetchall()

with open(args.output, "w") as fout:
    for r in results:
        fout.write("%s,%s,%s\n" % (r[0], r[1], r[2]))
