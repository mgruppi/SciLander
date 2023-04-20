import pickle
import random
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


from utils import get_labels, sanity_check

################################# GLOBAL #################################
data_dir = str(Path.home()) + '/data/nela/'
labels_file = '../data/labels_all.csv'
source_typos_file = '../data/source_typos.csv'
path_embeddings = "../results/content-sharing+semantic+stance.model"

TOTAL_MONTHS = 18
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
#########################################################################

################################ TRIPLET ################################
def prepare_triplets(embeddings, leave_out_sources, months):

    embeddings = embeddings[~embeddings['source'].isin(leave_out_sources)]

    triplets = pd.concat([pd.read_csv(f) for f in ['../data/triplets/content-sharing.csv', '../data/triplets/shift.csv', '../data/triplets/stance_triplets.csv']])

    #fix typos in sources
    source_fixes = pd.read_csv(source_typos_file, index_col='old')
    triplets = triplets.applymap(lambda s: source_fixes['new'].get(s, s))

    #create sources index only with sources contained in triplets
    sources_index = pd.read_csv(labels_file, usecols=['source']).drop_duplicates(subset='source').reset_index(drop=True)
    sources_index = sources_index[sources_index['source'].isin(np.unique(triplets.values))]
    sources_index = sources_index.reset_index(drop=True).reset_index().set_index('source')

    #remove leave-out sources and malformed triplets
    triplets = triplets[(triplets['a'].isin(leave_out_sources) | triplets['p'].isin(leave_out_sources) | triplets['n'].isin(leave_out_sources))]
    triplets = triplets[(triplets['a'] != triplets['p']) & (triplets['a'] != triplets['n']) & (triplets['p'] != triplets['n'])]

    #filter by months
    triplets = pd.concat([triplets[((triplets['a'] == s) | (triplets['p'] == s) | (triplets['n'] == s))].sample(frac=months/TOTAL_MONTHS, random_state=42) for s in leave_out_sources])

    #dictionarize triplets
    triplets = triplets[['a','p','n']].applymap(lambda x: sources_index['index'].get(x, np.nan)).dropna().astype({'a': int, 'p': int, 'n': int})
    sources_index = sources_index.reset_index().set_index('index')

    sources_index['leave_out'] = sources_index['source'].isin(leave_out_sources)


    return triplets, sources_index, embeddings
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
    def __init__(self, sources_index, embedding_dim, margin, metric, pretrained_embeddings):
        super(TripletsModel, self).__init__()
        self.sources_index = sources_index
        self.margin = margin
        self.metric = metric

        #create embeddings index for sources
        embeddings_index = pd.concat([sources_index[sources_index['leave_out'] == leave_out].reset_index().reset_index().set_index('index').rename(columns={'level_0': 'embeddings_index'})[['embeddings_index']] for leave_out in [True, False]])
        self.sources_index = sources_index.join(embeddings_index)

        #create two embeddings tensors
        self.golden_embeddings = nn.Embedding((sources_index['leave_out'] == False).sum(), embedding_dim)
        self.golden_embeddings.weight = nn.Parameter(torch.from_numpy(pretrained_embeddings['embedding'].apply(pd.Series).values), requires_grad=False)
        self.embeddings = nn.Embedding((sources_index['leave_out'] == True).sum(), embedding_dim, scale_grad_by_freq=True)


    def forward(self, a, p, n):
        if self.metric == 'euclidean':
            triplet_loss = nn.TripletMarginLoss(margin=self.margin)
        elif self.metric == 'cosine':
            triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y), margin=self.margin)
        
        #retrieve the appropriate embeddings tensor
        retrieve_embeddings = lambda x: self.golden_embeddings(torch.tensor(self.sources_index.loc[x]['embeddings_index'])) if self.sources_index.loc[x]['leave_out'] == False else self.embeddings(torch.tensor(self.sources_index.loc[x]['embeddings_index']))
        emb = {t: torch.stack(pd.DataFrame(t).astype(int)[0].apply(retrieve_embeddings).tolist()) for t in [a, p, n]}
        return triplet_loss(emb[a], emb[p], emb[n])


    def get_embeddings(self):
        source_embeddings = []
        source_embeddings += [self.sources_index[self.sources_index['leave_out'] == False].reset_index().merge(pd.DataFrame(self.golden_embeddings.weight.detach()).astype(float).apply(list, axis=1).reset_index(), left_on='embeddings_index', right_on='index', how='left').rename(columns={'index_x': 'index', 0: 'embedding'}).drop(['embeddings_index', 'index_y'], axis=1)]
        source_embeddings += [self.sources_index[self.sources_index['leave_out'] == True].reset_index().merge(pd.DataFrame(self.embeddings.weight.detach()).astype(float).apply(list, axis=1).reset_index(), left_on='embeddings_index', right_on='index', how='left').rename(columns={'index_x': 'index', 0: 'embedding'}).drop(['embeddings_index', 'index_y'], axis=1)]
        source_embeddings = pd.concat(source_embeddings).set_index('index').sort_index()
        return source_embeddings

#train source embeddings with triplet loss
def train_embeddings(triplets, sources_index, pretrained_embeddings,
                     embedding_dim,
                     lr,
                     epochs,
                     batch_size,
                     margin,
                     metric,
                     use_cuda=False,
                     device="cuda:0"):
    
    model = TripletsModel(sources_index, embedding_dim, margin, metric, pretrained_embeddings)
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

def train_temporal_embeddings():
    with open('../model/SciLander.emb', "rb") as fin:
        embeddings = pickle.load(fin)
    
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    
    result = []
    for _, leave_out_sources in kf.split(embeddings['source']):
        for months in [.5, 1, 2, 3, 6, 12, 18]:
            triplets, sources_index, pretrained_embeddings = prepare_triplets(embeddings, embeddings['source'][leave_out_sources], months=months)
            source_embeddings = train_embeddings(triplets, sources_index, pretrained_embeddings, embedding_dim=50, lr=1e-2, epochs=20, batch_size=1024, margin=1, metric='cosine')

            source_embeddings = get_labels(source_embeddings)

            X_train = source_embeddings[source_embeddings['leave_out'] == False]['embedding'].apply(pd.Series).values
            y_train = source_embeddings[source_embeddings['leave_out'] == False]['label']
            X_test = source_embeddings[source_embeddings['leave_out'] == True]['embedding'].apply(pd.Series).values
            y_test = source_embeddings[source_embeddings['leave_out'] == True]['label']
            model = KNeighborsClassifier(n_neighbors=30, metric='cosine')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            result += [[months, round(f1_score(y_test, y_pred), 3)]]

    pd.DataFrame(result).to_csv('../results/online_classification.csv', index=False)

def plot_f1_trimester():
    data = pd.read_csv('../results/online_classification.csv', names=['Month(s) of Publishing Activity', 'F1'])
    #boxplot
    sns.set(context='paper', style='white', color_codes=True, font_scale=3.5)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    sns.boxplot(x='Month(s) of Publishing Activity', y='F1', data=data, whis=[0, 100], palette="PuBu", ax=ax)
    sns.despine(left=True, bottom=True)
    plt.xticks(ticks=ax.get_xticks(), labels=['1/2', '1', '2', '3', '6', '12', '18'])
    plt.tight_layout()
    plt.savefig('../results/figures/online_classification.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # train_temporal_embeddings()
    plot_f1_trimester()
