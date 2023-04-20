import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.semi_supervised import LabelPropagation
from torch.utils.data import DataLoader, Dataset

from utils import get_labels, sanity_check

################################# GLOBAL #################################
data_dir = str(Path.home()) + '/data/nela/'
labels_file = '../data/labels_all.csv'
source_typos_file = '../data/source_typos.csv'
path_embeddings = "../results/content-sharing+semantic+stance.model"

torch.manual_seed(42)
#########################################################################

################################ TRIPLET ################################
def prepare_triplets(triplet_dfs, triplet_threshold):
    """
    Given a collectio nof triplet DataFrames, prepare them for training
    Args:
        triplet_dfs: (list) Pandas DataFrames with triplets. If None, then try to read files.

    Returns:
        triplets, source_index - The triplets and the source_index to use in training

    """    
    #aggregate all triplets
    triplets = pd.concat(triplet_dfs)

    #fix typos in sources
    source_fixes = pd.read_csv(source_typos_file, index_col='old')
    triplets = triplets.applymap(lambda s: source_fixes['new'].get(s, s))

    #filter rare triplets
    f = triplets[['a', 'p', 'n']].value_counts().reset_index()
    triplets = f[f[0] > triplet_threshold].merge(triplets, on=['a', 'p', 'n']).drop(columns=0)

    #create sources index only with sources contained in triplets
    sources_index = pd.read_csv(labels_file, usecols=['source']).drop_duplicates(subset='source').reset_index(drop=True)
    sources_index = sources_index[sources_index['source'].isin(np.unique(triplets.values))]
    sources_index = sources_index.reset_index(drop=True).reset_index().set_index('source')

    #dictionarize triplets
    triplets = pd.concat([triplets[['a','p','n']].applymap(lambda x: sources_index['index'].get(x, np.nan)), triplets['indicator']], axis=1).dropna().astype({'a': int, 'p': int, 'n': int})
    sources_index = sources_index.reset_index().set_index('index')

    #sanity check
    # sanity_check(triplets, sources_index)

    return triplets, sources_index
#########################################################################

############################### EMBEDDING ###############################

#triplet loss dataset and model classes
class TripletsDataset(Dataset):
    def __init__(self, triplets, to_gpu=False):
        self.triplets = triplets
        self.to_gpu = to_gpu

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        if self.to_gpu is False:
            t = self.triplets.iloc[idx]
            return (t['a'], t['p'], t['n'])
        else:
            t = self.triplets[idx]
            return t[0], t[1], t[2]


class TripletsModel(nn.Module):
    def __init__(self, sources_index, embedding_dim, margin, metric):
        super(TripletsModel, self).__init__()
        self.sources_index = sources_index
        self.margin = margin
        self.metric = metric
        self.embeddings = nn.Embedding(len(sources_index), embedding_dim, scale_grad_by_freq=True)

    def forward(self, a, p, n):
        if self.metric == 'euclidean':
            triplet_loss = nn.TripletMarginLoss(margin=self.margin)
        elif self.metric == 'cosine':
            triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y), margin=self.margin)
        return triplet_loss(self.embeddings(a), self.embeddings(p), self.embeddings(n))
        
    def get_embeddings(self):
        source_embeddings = self.sources_index.reset_index().merge(pd.DataFrame(self.embeddings.weight.detach()).astype(float).apply(list, axis=1).reset_index(), left_on='index', right_on='index', how='left').rename(columns={0: 'embedding'}).set_index('index')
        return source_embeddings

#train source embeddings with triplet loss
def train_embeddings(triplets, sources_index,
                     embedding_dim,
                     lr,
                     epochs,
                     batch_size,
                     margin,
                     metric,
                     use_cuda=False,
                     device="cuda:0"):
    
    model = TripletsModel(sources_index, embedding_dim, margin, metric)
    if use_cuda:
        model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    dataset = TripletsDataset(triplets, to_gpu=use_cuda)

    if use_cuda:
        dataset.triplets = torch.tensor(triplets[["a", "p", "n"]].to_numpy(dtype=int)).to(device)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    min_loss = 100.0
    for epoch in range(epochs):
        for batch in data_loader:
            a, p, n = batch
            optimizer.zero_grad()
            L = model(a, p, n)
            L.backward()
            optimizer.step()
            if L.item() < min_loss:
                source_embeddings = model.get_embeddings()
                min_loss = L.item()
        print('epoch:', epoch, 'loss:', L.item())

    # source_embeddings = model.get_embeddings()
    return source_embeddings
#########################################################################

#propagate labels to unlabeled sources
def label_propagation(golden_ratio = .2):
    with open(path_embeddings, "rb") as fin:
        source_embeddings = pickle.load(fin)
    
    #get label and indicate golden sources
    source_embeddings = get_labels(source_embeddings)    
    source_embeddings['golden'] = False
    golden_indices = sum([pd.Series(source_embeddings[source_embeddings['label'] == label].index).sample(frac=golden_ratio, random_state=42).to_list() for label in [0, 1]], [])
    source_embeddings.loc[golden_indices, 'golden'] = True

    X = source_embeddings['embedding'].apply(pd.Series).values
    y = source_embeddings.apply(lambda s: (-1 if s['golden'] == False else s['label']), axis=1)
    
    lp = LabelPropagation(gamma=.1, max_iter=int(1e+5))
    lp.fit(X, y)
    
    y_true = source_embeddings['label']
    y_pred = lp.predict(X)

    acc = (source_embeddings.loc[(y_true == y_pred)]['golden'] == False).sum() / (source_embeddings['golden'] == False).sum()
    print('Accuracy:', acc)

def plot_embeddings(labels_num):
    with open(path_embeddings, "rb") as fin:
        source_embeddings = pickle.load(fin)
    sns.set(context='paper', style='white', color_codes=True)
    source_embeddings = get_labels(source_embeddings)
    source_embeddings = pd.concat([source_embeddings, pd.DataFrame(TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(source_embeddings['embedding'].apply(pd.Series).values))], axis=1)
    if labels_num == 2:
        source_embeddings.loc[:, 'label'] = source_embeddings.apply(lambda s: 'reliable' if s['label'] == 0 else 'unreliable', axis=1)
    elif labels_num == 3:
        source_embeddings.loc[:, 'label'] = source_embeddings.apply(lambda s: 'reliable' if s['label'] == 0 else 'conspiracy-pseudoscience' if s['label'] == 1 else 'questionable-source', axis=1)    
    sns.scatterplot(data=source_embeddings, x=0, y=1, hue='label', palette='colorblind')
    plt.show()

def sources_algebra(source_1, sign, source_2):
    with open(path_embeddings, "rb") as fin:
        source_embeddings = pickle.load(fin)

    source_1_emb = np.array(source_embeddings.loc[source_embeddings['source']==source_1]['embedding'].tolist())
    source_2_emb = np.array(source_embeddings.loc[source_embeddings['source']==source_2]['embedding'].tolist())
    if sign == '+':
        query = source_1_emb + source_2_emb
    elif sign == '-':
        query = source_1_emb - source_2_emb

    max_similarity = 0.0
    for _, r in source_embeddings.iterrows():
        sim = cosine_similarity(np.array(r['embedding']).reshape(1, -1), query)[0][0]
        if sim > max_similarity:
            result = r['source']
            max_similarity = sim
    print(source_1, sign, source_2, '=', result)

if __name__ == "__main__":
    # triplets, sources_index = prepare_triplets()
    # source_embeddings = train_embeddings(triplets, sources_index)
    label_propagation(golden_ratio=.05)
    plot_embeddings(labels_num=3)
    sources_algebra('infowars', '+', 'veteranstoday')
