import pandas as pd

path_labels = "source_labels.csv"

df = pd.read_csv(path_labels)

print(df)
print(df.columns)

print("Conspiracy", sum(~(df["labels.conspiracy"].isna() & df["labels.pseudoscience"].isna())))

print(df["labels.factuality"].value_counts())
print("Reliable", sum(df["labels.factuality"] > 2))
print("Unreliable", sum(df["labels.factuality"] <= 2))
print("Partisan", sum(~df["labels.bias"].isna()))
