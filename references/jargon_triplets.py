import argparse
import random
from pathlib import Path

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from utils import clean_reference

################################# GLOBAL #################################
data_dir = str(Path.home()) + '/data/nela/'
source_sci_references_file = data_dir + 'source_sci_references.csv'
jargon_triplets_file = '../data/triplets/jargon_triplets.csv'
article_duplicates_file = data_dir + 'input/duplicates/article_duplicates.csv'
article_ids_file = data_dir + 'input/duplicates/article_ids.csv'
CDC_keywords_file = '../data/CDC+COVID_vocab.txt'
stopwords_file = '../data/stopwords_english.txt'

#parallelization
pandarallel.initialize(progress_bar=False, use_memory_fs=False, verbose=1)
tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument('--pos_threshold', type=int, default=0, help='Threshold for the pos_distance of the jargon of the sources.')
parser.add_argument('--neg_threshold', type=int, default=10, help='Threshold for the neg_distance of the jargon of the sources.')
parser.add_argument('--triplet_threshold', type=int, default=800, help='Threshold for rare triplets.')
parser.add_argument('--negative_sampling', type=str, default='False', help='Negative sampling for triplets.')
args = parser.parse_args()
pos_distance_threshold = args.pos_threshold
neg_distance_threshold = args.neg_threshold
triplet_threshold = args.triplet_threshold
negative_sampling = eval(args.negative_sampling)

MIN_PARAGRAPH_LENGTH = 128
##########################################################################

######################### TRIPLET PREPROCESSING #########################

#extract the initial set of triplets
def extract_jargon_triplets():
    source_sci_references = pd.read_csv(source_sci_references_file).dropna(subset=['paragraph'])
    source_sci_references = source_sci_references[source_sci_references['paragraph'].str.len() > MIN_PARAGRAPH_LENGTH]
    # source_sci_references = source_sci_references[source_sci_references['is_duplicate'] == False]
    source_sci_references['clean_reference'] = clean_reference(source_sci_references['reference'])
    
    # compute jargon ratio based on CDC keywords
    CDC_keywords = set(pd.read_csv(CDC_keywords_file, header=None)[0].str.lower())
    stopwords = set(pd.read_csv(stopwords_file, header=None)[0].str.lower())
    source_sci_references['jargon'] = source_sci_references['paragraph'].progress_apply(lambda p: sum([k in p.lower() for k in CDC_keywords]))

    #apply function
    def _extract(sources):
        #aggregate jargon per source
        sources = sources.groupby('source')[['jargon']].median().reset_index()

        #distinct sources needed for triplets
        if len(sources) < 3:
            return []

        #traverse sources list in both ways to find candidate triplets
        triplets = []
        for s in [sources.sort_values('jargon', ascending=True), sources.sort_values('jargon', ascending=False)]:
            for j in range(len(s) - 1):
                for w in range(j):
                    a = s.iloc[j]
                    p = s.iloc[w]
                    n = s.iloc[-1]
                    triplets += [{'a': a['source'], 'p': p['source'], 'n': n['source'], 'pos_distance': abs(a['jargon'] - p['jargon']), 'neg_distance': abs(a['jargon'] - n['jargon'])}]

        return triplets

    #extract candidate triplets from common references
    jargon_triplets = source_sci_references.groupby(['clean_reference'])[['source', 'jargon']].progress_apply(_extract).explode().dropna().apply(pd.Series)

    return jargon_triplets

#filter and clean triplets
def refine_jargon_triplets(jargon_triplets):    
    #select triplets based on their distances
    jargon_triplets = jargon_triplets[jargon_triplets['pos_distance'] <= pos_distance_threshold][['p', 'a']].merge(jargon_triplets[jargon_triplets['neg_distance'] >= neg_distance_threshold][['n', 'a']])

    #filter rare triplets
    f = jargon_triplets[['a', 'p', 'n']].value_counts().reset_index()
    jargon_triplets = f[f[0] > triplet_threshold].merge(jargon_triplets, on=['a', 'p', 'n']).drop(columns=0)

    if negative_sampling:
        #negative sampling from the complement of a pool of positive sources
        pool_p = jargon_triplets.groupby('a')['p'].apply(set)
        pool_n = jargon_triplets.groupby('a')['n'].apply(set)
        pool = set(pd.concat([jargon_triplets['a'], jargon_triplets['p'], jargon_triplets['n']]))
        while True:
            cur_pool_p = pool_p
            pool_p = pool_p.apply(lambda s: set([p for e in s for p in list(pool_p.get(e, []))+[e]]))
            if (cur_pool_p == pool_p).all(): break
        pool = (pool - pool_p).fillna(pool_n)
        jargon_triplets.loc[:, 'n'] = jargon_triplets.apply(lambda t, _=random.seed(42): random.choice(sorted(list(pool[t['a']]))), axis=1)

    #drop duplicated triplets
    jargon_triplets = jargon_triplets.drop_duplicates()

    return jargon_triplets
#########################################################################

#dump jargon triplets to a csv
def dump_jargon_triplets():
    jargon_triplets = extract_jargon_triplets()
    jargon_triplets = refine_jargon_triplets(jargon_triplets)
    jargon_triplets.to_csv(jargon_triplets_file, index=False)


if __name__ == "__main__":
    dump_jargon_triplets()
