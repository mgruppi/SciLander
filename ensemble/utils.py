import time

import numpy as np
import pandas as pd
from numpy import negative
from sklearn.metrics import roc_auc_score
import pickle


#remove aggregators from triplets
def remove_aggregators_triplets(triplets_file):
    triplets = pd.read_csv(triplets_file)
    aggregators = ['drudgereport', 'whatreallyhappened', 'therussophileorg']

    #fix typos in sources
    source_fixes = pd.read_csv('../data/source_typos.csv', index_col='old')
    triplets = triplets.applymap(lambda s: source_fixes['new'].get(s, s))

    #remove aggregators and malformed triplets
    triplets = triplets[~(triplets['a'].isin(aggregators) | triplets['p'].isin(aggregators) | triplets['n'].isin(aggregators))]
    triplets = triplets[(triplets['a'] != triplets['p']) & (triplets['a'] != triplets['n']) & (triplets['p'] != triplets['n'])]

    triplets.to_csv(triplets_file, index=False)

#remove aggregators from embeddings
def remove_aggregators_embeddings(embeddings_file):
    aggregators = ['drudgereport', 'whatreallyhappened', 'therussophileorg']
    with open(embeddings_file, "rb") as fin:
        embeddings = pickle.load(fin)

    embeddings = embeddings[~embeddings['source'].isin(aggregators)]

    with open(embeddings_file, "wb") as fout:
        pickle.dump(embeddings, fout)


#add labels to sources
def get_labels(references, labels_num=2):
    source_labels = pd.read_csv('../data/labels_all.csv').drop_duplicates(subset='source')
    if labels_num == 2:
        source_labels['label'] = source_labels['questionable-source'] | source_labels['conspiracy-pseudoscience']
    elif  labels_num == 3:
        source_labels['label'] = (source_labels['bias'] == 'conspiracy-pseudoscience')*1 + (source_labels['bias'] == 'questionable-source')*2
    references = references.merge(source_labels[['source', 'label']], on='source', how='inner')
    return references

#sanity check using the labels
def sanity_check(triplets, sources_index, metric='AUROC'):
    labels = get_labels(sources_index)

    if metric == 'ACC':
        triplets = triplets[['a', 'p', 'n']].applymap(lambda t: labels.loc[t]['label'])
        print('positive accuracy:', (triplets['a'] == triplets['p']).sum() / len(triplets))
        print('negative accuracy:', (triplets['a'] != triplets['n']).sum() / len(triplets))
        print('global accuracy:', ((triplets['a'] == triplets['p']) & (triplets['a'] != triplets['n'])).sum() / len(triplets))
    
    elif metric == 'P+R':
        for indicator in triplets['indicator'].drop_duplicates():
            print ('Indicator:', indicator)

            indicator_triplets = triplets[triplets['indicator'] == indicator][['a', 'p', 'n']]
            indicator_triplets['a'] = indicator_triplets['a'].apply(lambda t: labels.loc[t]['label'])
            
            for pair_type in ['positive', 'negative']:
                print('Pair type:', pair_type)
                pairs = indicator_triplets.groupby('a')[pair_type[0]].apply(list)

                for l in labels['label'].drop_duplicates().sort_values():
                    if pair_type == 'positive':
                        precision = pd.Series(pairs[l]).apply(lambda t: labels.loc[t]['label']).value_counts().get(l, 0)/len(pairs[l])
                        recall = pd.Series(pairs[l]).drop_duplicates().apply(lambda t: labels.loc[t]['label']).value_counts().get(l, 0)/(labels['label'] == l).sum()
                    elif pair_type == 'negative':
                        precision = pd.Series(pairs[l]).apply(lambda t: labels.loc[t]['label']).value_counts().get(1-l, 0)/len(pairs[l])
                        recall = pd.Series(pairs[l]).drop_duplicates().apply(lambda t: labels.loc[t]['label']).value_counts().get(1-l, 0)/(labels['label'] == 1-l).sum()

                    print('Label:', l, '\tPrecision:', precision, '\tRecall:', recall)

            print('#sources:', len(labels))
    elif metric == 'AUROC':
        for indicator in triplets['indicator'].drop_duplicates():
            print ('Indicator:', indicator)
            indicator_triplets = triplets[triplets['indicator'] == indicator][['a', 'p', 'n']]
            indicator_triplets = indicator_triplets[['a', 'p', 'n']].applymap(lambda t: labels.loc[t]['label'])

            print('positive AUROC:', roc_auc_score(indicator_triplets['a'], indicator_triplets['p']))
            print('negative AUROC:', roc_auc_score(indicator_triplets['a'], 1-indicator_triplets['n']))
            print('overall AUROC:', roc_auc_score(indicator_triplets['a'], indicator_triplets.apply(lambda t: t['a'] if (t['a'] == t['p']) & (t['a'] != t['n']) else 1-t['a'], axis=1)))
            print('#sources:', len(np.unique(triplets[triplets['indicator'] == indicator][['a', 'p', 'n']])))


def timed(func):
    """
    Decorator to time stuff.
    :param func: function to be called
    :return:
    """
    def timed_func(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print(" - Function %s - %.2f seconds" % (func.__name__, t1-t0))
        return result
    return timed_func

