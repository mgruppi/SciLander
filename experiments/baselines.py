import argparse
import pickle
import re
import sqlite3
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from simpletransformers.config.model_args import ModelArgs
from simpletransformers.language_representation import RepresentationModel
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

################################# GLOBAL ################################
data_dir = str(Path.home()) + '/data/nela/'
db_dump = data_dir + 'input/db/nela-covid.db'
source_sci_references_file = data_dir+'source_sci_references.csv'
source_typos_file = '../data/source_typos.csv'
MAX_BERT_SEQUENCE = 32
#tokenizer regex
regex = re.compile( r'\d+(?:[\,\.](?:\d+))'  # Numbers and digits (including ',' and '.').
                    r'|(?:\w(?:\.\w)+(?:\.)?)'  # Acronyms (e.g.: U.S.)
                    r'|\w+(?:[-_]\w+)*'  # Words and bi-grams separated by -
                    r'|[\.\,\?\!\;]+'  # Matches punctuation
                    r'|[\@\#\$\%]+')  # Matches special symbols
################################# ###### ################################

parser = argparse.ArgumentParser()
#BERT Parameters
parser.add_argument('--use_cuda', type=str, default='False', help='Use cuda to compute the embeddings.')
parser.add_argument('--n_gpu', type=int, default=1, help='#GPUs used.')
#node2vec Parameters
parser.add_argument('--dim', type=int, default=128, help='node2vec dim.')
parser.add_argument('--workers', type=int, default=4, help='node2vec workers.')
parser.add_argument('--walk_length', type=int, default=30, help='node2vec walk_length.')
parser.add_argument('--num_walks', type=int, default=200, help='node2vec num_walks.')
#joint Parameters
parser.add_argument('--PCA_dim', type=int, default=300, help='PCA reduction dimension.')

args = parser.parse_args()
use_cuda = eval(args.use_cuda)
n_gpu = args.n_gpu
dim = args.dim
workers = args.workers
walk_length = args.walk_length
num_walks = args.num_walks
PCA_dim = args.PCA_dim

def BERT_baseline():
    #read df from db
    df = pd.read_sql_query('SELECT source, title, content FROM newsdata',con = sqlite3.connect(db_dump))

    #fix typos in sources
    source_fixes = pd.read_csv(source_typos_file, index_col='old')
    df['source'] = df['source'].apply(lambda s: source_fixes['new'].get(s, s))

    #concatenate title & content and limit #tokens
    df['content'] = (df['title'] + ' ' + df['content']).str.findall(regex).apply(lambda l: ' '.join(l[:MAX_BERT_SEQUENCE]))
    df = df.drop_duplicates()

    #compute BERT and SciBERT embeddings
    for model_name, embeddings_file in zip(['bert-base-uncased', 'allenai/scibert_scivocab_uncased'], ['../results/BERT_embeddings.model', '../results/SciBERT_embeddings.model']):
        model = RepresentationModel(model_type='bert', model_name=model_name, use_cuda=use_cuda, args = ModelArgs(max_seq_length=MAX_BERT_SEQUENCE, n_gpu=n_gpu))
        df['embedding'] = model.encode_sentences(df['content'], combine_strategy='mean', batch_size=1024).tolist()
        #aggregate for each source
        final_embeddings = df.groupby('source')['embedding'].apply(lambda emb: np.stack(emb).mean(axis=0)).reset_index()
        with open(embeddings_file, "wb") as fout:
            pickle.dump(final_embeddings.set_index('source'), fout)


def node2vec_baseline():
    source_sci_references = pd.read_csv(source_sci_references_file)

    #fix typos in sources
    source_fixes = pd.read_csv(source_typos_file, index_col='old')
    source_sci_references['source'] = source_sci_references['source'].apply(lambda s: source_fixes['new'].get(s, s))

    source_sci_references = source_sci_references.groupby('source')['clean_reference'].apply(lambda x: ' '.join(x)).reset_index()

    #compute tfidf
    vectorizer = CountVectorizer(token_pattern=r'\S+', binary=True)
    X = vectorizer.fit_transform(source_sci_references['clean_reference'].tolist())

    #transform to adjacency matrix
    Z = X @ X.T

    #create graph
    g = nx.from_scipy_sparse_matrix(Z)

    #run node2vec model
    node2vec = Node2Vec(g, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers, seed=42)
    model = node2vec.fit()

    #get embeddings
    embeddings = pd.concat([source_sci_references['source'], pd.Series(model.wv.vectors.tolist(), name='embedding')], axis=1)
    embeddings = embeddings[embeddings['source'].isin(pd.read_csv('../data/labels_all.csv')['source'])]

    with open('../results/node2vec_embeddings.model', "wb") as fout:
        pickle.dump(embeddings, fout)

def joint_baseline():
    for model in ['BERT.emb', 'SciBERT.emb']:
        with open('../model/'+ model, "rb") as fin:
            BERT_embeddings = pickle.load(fin)
        with open('../model/node2vec.emb', "rb") as fin:
            node2vec_embeddings = pickle.load(fin)

        #join & fill NaNs
        joint_embeddings = BERT_embeddings.merge(node2vec_embeddings, on='source', how='outer')
        joint_embeddings['embedding_x'] = joint_embeddings['embedding_x'].apply(lambda e: [.0]* len(BERT_embeddings['embedding'][0]) if e is np.nan else e)
        joint_embeddings['embedding_y'] = joint_embeddings['embedding_y'].apply(lambda e: [.0]* len(node2vec_embeddings['embedding'][0]) if e is np.nan else e)

        #concatenate embeddings
        joint_embeddings['embedding'] = joint_embeddings.apply(lambda e: list(e['embedding_x']) + list(e['embedding_y']), axis=1)

        #dimensionality reduction
        joint_embeddings['embedding'] = PCA(n_components=PCA_dim).fit_transform(np.stack(joint_embeddings['embedding'])).tolist()

        with open('../model/joint_' + model, "wb") as fout:
            pickle.dump(joint_embeddings[['source', 'embedding']], fout)


if __name__ == "__main__":
    # BERT_baseline()
    # node2vec_baseline()
    joint_baseline()

