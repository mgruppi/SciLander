#!/bin/bash

path_triplets="../data/triplets/"
signals=("shift" "content-sharing" "jargon_triplets" "stance_triplets")

dim=50
metric="cosine"
margin=1
epochs=50

for s in "${signals[@]}"
do
  echo "$s"
  python3 train_source_embeddings.py \
          --margin="$margin"  \
          --metric="$metric"  \
          --use-cuda          \
          --dim="$dim"        \
          --epochs="$epochs"  \
          --output="../model/$s.emb" \
          --indicators="$s"
done

 Run once with all signals

python3 train_source_embeddings.py \
        --margin="$margin"  \
        --metric="$metric"  \
        --use-cuda          \
        --dim="$dim"        \
        --output="../model/content-sharing+shift+stance+jargon.emb"  \
        --indicators "${signals[@]}"


signals_3=("shift" "content-sharing" "stance_triplets")
python3 train_source_embeddings.py \
        --margin="$margin"  \
        --metric="$metric"  \
        --use-cuda          \
        --dim="$dim"        \
        --output="../model/content-sharing+shift+stance.emb"  \
        --indicators "${signals_3[@]}"
