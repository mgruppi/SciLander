#!/bin/bash

# COMPARE CORD-19 TO NEWS AND COCA
# embeddings=("../data/embeddings/coca/coca_all.vec" "../data/embeddings/nela_covid_0.vec" "../data/embeddings/nela_covid_2.vec")
embeddings="../data/embeddings/nela-groups"
cord_embeddings="../data/embeddings/cord19/cord19.vec"
# frequencies=("../data/coca/frequency.txt" "../data/nela/freq_nela_covid_0.txt" "../data/nela/freq_nela_covid_2.txt")
vocab_file="data/CDC+COVID_vocab.txt"
k=1
# for ((i=0; i<"${#embeddings[@]}"; ++i));
for emb_file in "$embeddings"/*
do
  echo "Running " "$emb_file"
  python3 mapping.py "$cord_embeddings" "$emb_file" \
                  --freq_a=../data/cord-19/freq_cord-19.txt \
                  --freq_b=../data/coca/frequency.txt \
                  --vocab="$vocab_file" \
                  --k=$k \
                  --non_matching

  # Run in the other direction
  python3 mapping.py "$emb_file" "$cord_embeddings" \
                  --freq_a=../data/coca/frequency.txt \
                  --freq_b=../data/cord-19/freq_cord-19.txt \
                  --vocab="$vocab_file" \
                  --k=$k \
                  --non_matching
done
