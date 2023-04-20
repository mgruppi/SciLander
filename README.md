# SciLander: Mapping the Scientific News Landscape

## Repository Structure

- **Data**: The data used in this study includes the computed triplets, the vocabulary files, and the source labels. The corpus is not provided directly in this repository but is openly accessible (see the Corpus section for more details).
- **Indicators**: Code for individually computing the four indicators can be found in the folders: `content_sharing_network`, `references`, and `semantic_shift`.
- **Ensemble**: Code for training unsupervised source embeddings using the indicators.
- **Experiments**: All the experiments we conducted for our submission to ICWSM 2023.
- **Results**: Plots of the experiments we conducted for our submission to ICWSM 2023.
- **Model**: SciLander models as well as baselines models that we used in our experiments.


## Setup


Set up the environment using Python VirtualEnv. From the root directory, run:

```
python -m venv venv/
```

Activate the environment just created:
```
source venv/bin/activate
```

Install the dependencies:
```
pip install -r requirements.txt
```

## Corpus

The corpus used in this work was a combination of the [NELA-GT-2020](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CHMUYZ) and [NELA-GT-2021](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/RBKVBM) datasets, which can be downloaded in SQLite and JSON formats.

We filtered the corpus to retrieve only articles related to COVID-19 using a keyword matching procedure. The keywords can be found in `data/CDC+COVID_vocab.txt`. We selected all articles for which the `title` OR `content` had a match with at least one keyword from the list.

The SQLite database can be converted into CSV by using the script `preprocessing/nela_to_csv.py`.


## Training a triplet loss model

Train a triplet loss using the triplets in `data/triplets` using the escript `ensemble/train_source_embeddings.py`.

```
cd ensemble
python train_source_embeddings.py
```


## Computing triplets

Pre-computed triplets are provided in `data/triplets`. If you want to compute your own triplets, you can do so using the following scripts:

- __Jargon triplets__: Use `references/jargon_triplets.py`.
- __Stance triplets__: Use `references/stance_triplets.py`.
- __Content sharing triplets__: Use `content_sharing_network/csn_features.py`.
- __Semantic shift triplets__: Use `semantic_shift/semantic_shift_triplets.py`.


## Pre-trained models

The pre-trained source embeddings models can be found in directory `model`. Look for any file with extension `.emb`.


## Experiments

Most experiments require the pre-trained source embedding models from found in the `model` folder, in addition to the source labels found in the `data` folder.
