import argparse
import random
from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
from transformers import pipeline

from utils import clean_reference

################################# GLOBAL #################################
data_dir = str(Path.home()) + '/data/nela/'
source_sci_references_file = data_dir + 'source_sci_references.csv'
stance_triplets_file = '../data/triplets/stance_triplets.csv'
article_duplicates_file = data_dir + 'input/duplicates/article_duplicates.csv'
article_ids_file = data_dir + 'input/duplicates/article_ids.csv'

#parallelization
pandarallel.initialize(progress_bar=False, use_memory_fs=False, verbose=1)
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument('--pos_threshold', type=float, default=.0, help='Threshold for the pos_distance of the stance of the sources.')
parser.add_argument('--neg_threshold', type=float, default=.99, help='Threshold for the neg_distance of the stance of the sources.')
parser.add_argument('--triplet_threshold', type=int, default=30, help='Threshold for rare triplets.')
parser.add_argument('--negative_sampling', type=str, default='True', help='Negative sampling for triplets.')
args = parser.parse_args()
pos_distance_threshold = args.pos_threshold
neg_distance_threshold = args.neg_threshold
triplet_threshold = args.triplet_threshold
negative_sampling = eval(args.negative_sampling)

#transformers
MIN_PARAGRAPH_LENGTH = 128
MAX_PARAGRAPH_LENGTH = 1024
#select GPU device; -1 disables GPU and computes everything on CPU 
GPU_DEVICE = -1
stance_labels=['negative']
hypothesis_template = "This example is {}."
##########################################################################

######################### TRIPLET PREPROCESSING #########################

#extract the initial set of triplets
def extract_stance_triplets():
    source_sci_references = pd.read_csv(source_sci_references_file).dropna(subset=['stance'])
    # source_sci_references = source_sci_references[source_sci_references['is_duplicate'] == False]
    source_sci_references['clean_reference'] = clean_reference(source_sci_references['reference'])

    #apply function
    def _extract(sources):
        #aggregate stance per source
        sources = sources.groupby('source')[['stance']].median().reset_index()

        #distinct sources needed for triplets
        if len(sources) < 3:
            return []

        #traverse sources list in both ways to find candidate triplets
        triplets = []
        for s in [sources.sort_values('stance', ascending=True), sources.sort_values('stance', ascending=False)]:
            for j in range(len(s) - 1):
                for w in range(j):
                    a = s.iloc[j]
                    p = s.iloc[w]
                    n = s.iloc[-1]
                    triplets += [{'a': a['source'], 'p': p['source'], 'n': n['source'], 'pos_distance': abs(a['stance'] - p['stance']), 'neg_distance': abs(a['stance'] - n['stance']), 'anchor_stance': a['stance']}]

        return triplets

    #extract candidate triplets from common references
    stance_triplets = source_sci_references.groupby(['clean_reference'])[['source', 'stance']].progress_apply(_extract).explode().dropna().apply(pd.Series)

    return stance_triplets

#filter and clean triplets
def refine_stance_triplets(stance_triplets):
    #select triplets based on their distances
    stance_triplets = stance_triplets[stance_triplets['pos_distance'] <= pos_distance_threshold][['p', 'a']].merge(stance_triplets[stance_triplets['neg_distance'] >= neg_distance_threshold][['n', 'a']])

    #filter rare triplets
    f = stance_triplets[['a', 'p', 'n']].value_counts().reset_index()
    stance_triplets = f[f[0] > triplet_threshold].merge(stance_triplets, on=['a', 'p', 'n']).drop(columns=0)

    if negative_sampling:
        #negative sampling from the complement of a pool of positive sources
        pool_p = stance_triplets.groupby('a')['p'].apply(set)
        pool_n = stance_triplets.groupby('a')['n'].apply(set)
        pool = set(pd.concat([stance_triplets['a'], stance_triplets['p']]))
        while True:
            cur_pool_p = pool_p
            pool_p = pool_p.apply(lambda s: set([p for e in s for p in list(pool_p.get(e, []))+[e]]))
            if (cur_pool_p == pool_p).all(): break
        pool = (pool - pool_p).fillna(pool_n)
        stance_triplets.loc[:, 'n'] = stance_triplets.apply(lambda t, _=random.seed(42): random.choice(sorted(list(pool[t['a']]))), axis=1)

    return stance_triplets

def mark_duplicate_articles():
    article_duplicates = pd.read_csv(article_duplicates_file, names=['_', 'id'])[['id']].drop_duplicates()

    article_ids = pd.read_csv(article_ids_file, names=['id', 'url']).drop_duplicates()
    article_duplicates = article_duplicates.merge(article_ids)[['url']]
    article_duplicates['is_duplicate'] = True

    source_sci_references = pd.read_csv(source_sci_references_file)
    source_sci_references = source_sci_references.merge(article_duplicates, how='left')
    source_sci_references['is_duplicate'] = source_sci_references['is_duplicate'].fillna(False)

    source_sci_references.to_csv(source_sci_references_file, index=False)
#########################################################################

#compute paragraph stance using zero-shot classification
def compute_paragraph_stance():
    source_sci_references = pd.read_csv(source_sci_references_file)
    stance_pipeline =  pipeline('zero-shot-classification', model='facebook/bart-large-mnli', device=GPU_DEVICE)
    zero_shot = lambda p : stance_pipeline(p, stance_labels, hypothesis_template=hypothesis_template)
    result_dict = lambda r: {l:s for l, s in zip(r['labels'], r['scores'])}
    aggegate_value = lambda d: 1 - d['negative']
    source_sci_references['stance'] = source_sci_references['paragraph'].progress_apply(lambda p: aggegate_value(result_dict(zero_shot(p[:MAX_PARAGRAPH_LENGTH]))) if len(str(p)) > MIN_PARAGRAPH_LENGTH else None)
    source_sci_references.to_csv(source_sci_references_file, index=False)

#dump stance triplets to a csv
def dump_stance_triplets():
    stance_triplets = extract_stance_triplets()
    stance_triplets = refine_stance_triplets(stance_triplets)
    stance_triplets.to_csv(stance_triplets_file, index=False)

if __name__ == "__main__":
    #compute_paragraph_stance()
    # mark_duplicate_articles()
    dump_stance_triplets()
