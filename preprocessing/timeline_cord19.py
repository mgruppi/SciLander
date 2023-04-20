"""
This script implements functions to analyze the CORD-19 data set in JSON format.
"""
import argparse
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from cord19_to_txt import read_metadata
from io import StringIO


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
        data = list()
        for row in reader:
            # Make sha column the article identifier (cord_uid)
            data.append(row)
    return data


def get_date(s):
    """
    Gets datetime object from date string in format YYYY-MM-DD or YYYY
    Returns datetime object
    """

    if len(s.split("-")) < 3:
        d = datetime.strptime(s, "%Y")
    else:
        d = datetime.strptime(s, "%Y-%m-%d")
    return d.strftime("%s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=str, help="Path to CORD-19 metadata")

    args = parser.parse_args()

    metadata = read_metadata(args.metadata)

    df_json = json.dumps(metadata)
    print(metadata[0].keys())
    df = pd.read_json(StringIO(df_json))  # Pandas no longer allows string JSON, so we use StringIO
    df['time'] = pd.to_datetime(df["publish_time"], format="%Y-%m-%d")
    print(df.columns)

    d_t = timedelta(weeks=1)
    t_0 = datetime.strptime("2020-01-01", "%Y-%m-%d")
    t_f = datetime.strptime("2020-11-01", "%Y-%m-%d")

    t = t_0
    x = list()
    y = list()
    while t < t_f:
        c = len(df[(df["time"] > t) & (df["time"] < t+d_t)])  # Count articles within d_t
        print(t.strftime("%Y-%m-%d"), c, sep=',')
        x.append(t.strftime("%Y-%m-%d"))
        y.append(c)
        t += d_t

    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
