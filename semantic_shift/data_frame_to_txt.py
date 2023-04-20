"""
Retrieves text from the `corpus.df` 
"""

import pandas as pd

sources_cluster_a = ["newswars", "davidicke", "newspunch", "infowars", "humansarefree",
                        "prisonplanet", "wakingtimes", "thedcclothesline"]
sources_cluster_b = ["mercola", "sanevax", "junksciencecom", "healthyholisticliving", "althealthworks",
                        "allianceadvancedhealth"]
sources_cluster_c = ["washingtonpost", "npr", "thehill", "vox", "usnews"]

source_clusters = [sources_cluster_a, sources_cluster_b, sources_cluster_c]
output_files = ["political_unreliable.txt", "health_conspiracy.txt", "mainstream.txt"]

df = pd.read_pickle('corpus/corpus.df')

for c, path in zip(source_clusters, output_files):
    df_c = df[df["source"].isin(c)]
    with open(path, "w") as fout:
        print(*df_c["content"], file=fout)


