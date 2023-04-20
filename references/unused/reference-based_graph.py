import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from utils import analyze_url, plot_communities

################################# GLOBAL ################################
data_dir = str(Path.home()) + '/data/nela/'
nlp = spacy.load('en_core_web_lg')

sources = pd.read_csv(data_dir+'sources_references.csv')
sources_labels = pd.read_csv(data_dir+'source_labels.csv')

covid_vocab = open(data_dir + 'COVID_vocab.txt').read().splitlines()+ open(data_dir + 'CDC_vocab.txt').read().splitlines()
covid_vocab = set([w.lower() for w in covid_vocab])

sentiment_pipeline = pipeline('sentiment-analysis')
SENTIMENT_THRESHOLD = .7
################################# ###### ################################

################################ HELPERS ################################

#Extract semantic information from text
def semantic_extraction(text, covid_vocab):
    
    text = ' '.join([str(t).lower() for t in text])
    sentiment = sentiment_pipeline(text[:512])[0]
    sentiment = 0 if sentiment['score'] < SENTIMENT_THRESHOLD else 1 if sentiment['label'] == 'POSITIVE' else -1 if sentiment['label'] == 'NEGATIVE' else np.NaN
    vocab = set(text.split(' ')).intersection(covid_vocab)
    text = nlp(text)
    entities = set([e.text.lower() for e in text.ents if e.label_ in ['PERSON', 'ORG', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'CARDINAL']])
    vocab = vocab.union(entities)
    vector = text.vector
    
    semantic_text = np.NaN if np.count_nonzero(vector) == 0 else (vocab, vector, sentiment)
    return semantic_text


#Compute semantic similarity
def semantic_similarity(semantic_text_1, semantic_text_2):
    vocab_1, vector_1, sentiment_1 = semantic_text_1
    vocab_2, vector_2, sentiment_2 = semantic_text_2

    sign = 2*int(np.sign(sentiment_1*sentiment_2) >= 0)-1
    vocab_sim = len(vocab_1.intersection(vocab_2)) + 1
    vec_sim = cosine_similarity(vector_1.reshape(1, -1), vector_2.reshape(1, -1))[0,0]
    return vocab_sim * vec_sim * sign

################################ ####### ################################

def tfidf_graph(sources):

    sources['clean_reference'] = sources['reference'].apply(lambda x: (lambda y: y['domain']+'.'+y['suffix']+y['path'])(analyze_url(x)))
    sources = sources.groupby('domain')['clean_reference'].apply(lambda x: ' '.join(x)).reset_index()

    #compute tfidf
    vectorizer = TfidfVectorizer(token_pattern=r'\S+')
    X = vectorizer.fit_transform(sources['clean_reference'].tolist())

    #transform to adjacency matrix
    Z = X @ X.T
    Z = Z.todense()
    np.fill_diagonal(Z, 0)


def tfidf_sim_graph(sources, covid_vocab):
    sources['clean_reference'] = sources['reference'].apply(lambda x: (lambda y: re.sub(r'[^\S]+', '', y['domain']+'.'+y['suffix']+y['path']).encode("ascii", errors="ignore").decode())(analyze_url(x)))

    #compute tfidf
    sources_tfidf = sources.groupby(['domain', 'clean_reference'])['paragraph'].count().reset_index()
    sources_tfidf.columns = ['domain', 'reference', 'count']
    sources_tfidf = sources_tfidf.pivot(index='domain', columns='reference', values='count')
    sources_tfidf = sources_tfidf.fillna(0)
    sources_tfidf = TfidfTransformer().fit_transform(sources_tfidf.values)
    
    #compute vector representation
    sources_sim = sources.groupby(['domain', 'clean_reference'])['paragraph'].apply(lambda l: semantic_extraction(l, covid_vocab)).reset_index()
 
    sources_sim.columns = ['domain', 'reference', 'semantic_text']
    sources_sim = sources_sim.pivot(index='domain', columns='reference', values='semantic_text')

    #custom matmul that we aggregate the similarity between two nodes
    sources_matrix = np.zeros((sources_sim.shape[0],sources_sim.shape[0]))

    for i,row in enumerate(sources_sim.values): 
        not_null_row = np.argwhere(~pd.isnull(row))
        for j,col in enumerate(sources_sim.values):
            if i<j:
                index_intersection = np.intersect1d(not_null_row, np.argwhere(~pd.isnull(col)),assume_unique=True)
                if len(index_intersection) != 0:
                    # min_similarity = 1.0
                    for w in index_intersection:
                        sources_matrix[i,j] += semantic_similarity(row[w], col[w]) * sources_tfidf[i, w] * sources_tfidf[j, w]
                        # sim = cosine_similarity(row[w].reshape(1, -1), col[w].reshape(1, -1)) * sources_tfidf[i, w] * sources_tfidf[j, w]
                        # if sim < min_similarity:
                        #     min_similarity = sim
                    # sources_matrix[i,j] = min_similarity

    np.save(data_dir + 'sources_matrix.npy', sources_matrix)


def find_similar_sources(threshold=0, k=10):
    sources_matrix = np.load(data_dir + 'sources_matrix.npy')

    G=nx.from_numpy_matrix(np.where(sources_matrix < threshold, 0, sources_matrix))
    G = nx.relabel_nodes(G, lambda x: sources['domain'].drop_duplicates().sort_values().iloc[x])


    similar_sources = [{n: [k+' ('+str('{0:.2f}'.format(v['weight']))+')' for k, v in sorted(dict(G[n]).items(), key=lambda x: x[1]['weight'], reverse=True)[:k]]} for n in G.nodes()]        
    similar_sources = pd.DataFrame(similar_sources)
    similar_sources = pd.DataFrame(np.diag(similar_sources), index=similar_sources.columns)[0].apply(pd.Series)
    similar_sources.to_csv(data_dir + 'similar_sources_refs.csv')

def plot_sources():
    similar_sources = pd.read_csv(data_dir + 'similar_sources_refs.csv').rename(columns={'Unnamed: 0': 'domain'})
    sources_labels = pd.read_csv(data_dir+'source_labels.csv')
    sources_labels['domain'] = sources_labels.source.apply(lambda s: re.search(s+r'[^\s]+', ' '.join(similar_sources.domain))).dropna().apply(lambda m: m.group(0))
    similar_sources = similar_sources.merge(sources_labels)

    nodes = similar_sources[['domain', 'label']]

    similar_sources = pd.concat([similar_sources[['domain', i]].rename(columns={i: 'edge'}).dropna() for i in similar_sources.columns[1:-2]])
    similar_sources['weight'] = similar_sources['edge'].apply(lambda x: float(re.sub(r'[\(\)]', '', re.search(r'\(.*\)', x).group(0)))) 
    similar_sources['edge'] = similar_sources['edge'].apply(lambda x: re.sub(r'\s.*$', '', x))
    similar_sources = similar_sources[similar_sources['edge'].isin(nodes['domain'])]

    similar_sources = similar_sources[similar_sources['weight'] > 0.01]

    similar_sources['domain'] = similar_sources['domain'].apply(lambda x: nodes.loc[nodes.domain == x].index[0])
    similar_sources['edge'] = similar_sources['edge'].apply(lambda x: nodes.loc[nodes.domain == x].index[0])

    similar_sources['weight'] = np.sqrt(similar_sources['weight'])

    G = nx.from_pandas_edgelist(similar_sources, source='domain', target='edge', edge_attr='weight', create_using=nx.DiGraph)
    G.add_nodes_from(nodes.index)

    pos = plot_communities(G, nodes['label'].to_dict())

    edge_colors = [e[2]['weight'] for e in G.edges(data=True)]

    nx.draw_networkx_nodes(G, pos, node_size=5, node_color=[{0: 'tab:orange', 1: 'tab:green', 2:'tab:purple'}[nodes.loc[n]['label']] for n in G.nodes()])
    edges = nx.draw_networkx_edges(G, pos, node_size=5, arrowstyle="->", arrowsize=10, edge_color=edge_colors, edge_cmap='PuBuGn', width=2)

    pc = mpl.collections.PatchCollection(edges, cmap='PuBuGn')
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    # tfidf_graph(sources)
    # tfidf_sim_graph(sources, covid_vocab)
    find_similar_sources(k=10)
    plot_sources()

