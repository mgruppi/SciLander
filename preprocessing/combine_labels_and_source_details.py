"""
Given a source label file (such as labels_all.csv) and a source detail JSON (such as source_details.json)
Combine both into a single, pandas-readable CSV format.
"""
import pandas as pd
import json


def fix_source_names(df, column_name, typos):
    """
    Fix source names using `typos` as the correct names.
    Applies to every source in DataFrame `df`.
    Args:
        df: Pandas DataFrame to correct.
        column_name: Name of the column containing the source name (str).
        typos: Dictionary mapping source name to correct name.

    Returns:
        df_fixed: DataFrame with correct names.
    """

    sources = df[column_name]
    sources_fixed = pd.Series([typos[s] if s in typos else s for s in sources])
    df[column_name] = sources_fixed

    return df


def main():
    path_labels = "../data/labels_all.csv"
    path_details = "../data/source_details.json"
    path_typos = "../data/source_typos.csv"
    path_output = "../data/source_labels.csv"

    df_labels = pd.read_csv(path_labels)

    with open(path_details) as fin:
        source_details = json.load(fin)

    df_details = pd.json_normalize(source_details)

    with open(path_typos) as fin:
        fin.readline()  # Read header
        source_typos = dict(map(lambda s: s.strip().split(",", 1), fin.readlines()))

    # Fix source names
    df_labels = fix_source_names(df_labels, "source", source_typos)
    df_details = fix_source_names(df_details, "name", source_typos)
    df_merged = df_labels.join(df_details.set_index("name"), on="source", how="left", lsuffix="_old")

    print(df_merged[df_merged["source"] == "cnn"])

    columns_to_drop = ["press_freedom", "factuality", "country_old", "questionable-source", "conspiracy-pseudoscience",
                       "label", "bias"]

    df = df_merged.drop(columns=columns_to_drop)
    df = df.drop_duplicates(["source"])

    # Fix category name from 'fake-news' to 'questionable-source' -
    # The name 'fake news' as attributed by MBFC but later changed to the latter.
    df["category"] = df["category"].str.replace("fake-news", "questionable-source")

    print(df.columns)

    print(df[df["labels.factuality"].isna()])
    df.to_csv(path_output, index=None)


if __name__ == "__main__":
    main()
