import pandas as pd
import plotly.express as px
from knn_classification import read_source_embeddings, read_labels
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os


path_embeddings = "../model/stance.emb"
path_labels = "../data/source_labels.csv"


model_name = os.path.basename(path_embeddings).split(".")[0]

np.random.seed(42)

index_to_source, source_to_index, embeddings = read_source_embeddings(path_embeddings)
label_data = read_labels(path_labels, index_to_source)

valid_labels = ~label_data["class"].isna()
label_data = label_data[valid_labels]
index_to_source = index_to_source[valid_labels]
embeddings = embeddings[valid_labels]

pca = True

if pca:
    x = PCA(n_components=2).fit_transform(embeddings)
else:
    x = TSNE(n_components=2, metric="cosine", square_distances=True).fit_transform(embeddings)

data = pd.DataFrame({"x": x[:, 0], "y": x[:, 1], "source": index_to_source})
data = data.join(label_data.set_index("source"), on="source", how="left", lsuffix="_old_")

color = "labels.conspiracy"
symbol = "category"

left_or_right = {"leftcenter": "left", "right-center": "right", "rightcenter": "right", "extremeright": "right"}
data["category"] = data["category"].apply(lambda s: left_or_right[s] if s in left_or_right else s)
data["category"] = data["category"].fillna("unknown")
data["labels.bias"] = data["labels.bias"].fillna("unknown")

fig = px.scatter(data, x="x", y="y", color=color,
                 hover_data=["source", "website", "class", "labels.bias", "labels.factuality", "category"])

fig.update_traces(marker=dict(size=14))
fig.update_layout(
    template="plotly_white"
)

fig.update_layout

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            buttons=list([
                dict(
                    args=["color", data["labels.factuality"]],
                    label="Factuality",
                    method="restyle"
                ),
                dict(
                    args=["color", "labels.factuality"],
                    label="Leaning",
                    method="relayout"
                )
            ])
        )
    ]
)

fig.update_layout(
    title_text=os.path.basename(path_embeddings),
)

fig.write_html('../results/interactive/%s' % os.path.basename(path_embeddings).replace(".emb", ".html"))

fig.show()

