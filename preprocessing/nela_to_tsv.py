"""
Reads NELA database and outputs tsv in SciLens format
Each line has triple
url\ttitle\tfull_text
"""
import sqlite3
import re
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to NELA database")
    parser.add_argument("output", type=str, help="Path to TSV output file")

    args = parser.parse_args()

    re_tab = re.compile("\t+|\n+")


    con = sqlite3.connect(args.input)

    query = "SELECT url, title, content FROM newsdata ORDER BY published_utc"

    r = con.cursor().execute(query)

    fout = open(args.output, "w")
    fout.write("url\ttitle\tfull_text\n")
    for row in r.fetchall():
        url = row[0]
        title = re_tab.sub(" ", row[1])
        full_text = re_tab.sub(" ", row[2])
        fout.write("%s\t%s\t%s\n" % (url, title, full_text))
    fout.close()



if __name__ == "__main__":
    main()
