import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns

content_triplets = pd.read_csv('../data/triplets/content-sharing.csv')
semantic_triplets = pd.read_csv('../data/triplets/shift.csv')
stance_triplets = pd.read_csv('../data/triplets/stance_triplets.csv')
jargon_triplets = pd.read_csv('../data/triplets/jargon_triplets.csv')
triplets_overlap = '../results/figures/triplets_overlap.pdf'
sources_overlap = '../results/figures/sources_overlap.pdf'

#triplets overlap
content_triplets_set = set(content_triplets['a']+content_triplets['p']+content_triplets['n'])
semantic_triplets_set = set(semantic_triplets['a']+semantic_triplets['p']+semantic_triplets['n'])
stance_triplets_set = set(stance_triplets['a']+stance_triplets['p']+stance_triplets['n'])
jargon_triplets_set = set(jargon_triplets['a']+jargon_triplets['p']+jargon_triplets['n'])

all_sets = [content_triplets_set, semantic_triplets_set, jargon_triplets_set, stance_triplets_set]
ticks = ['copy', 'shift', 'jargon', 'stance']
set_overlap = np.array(([[len(set1.intersection(set2))/len(set1) for set2 in all_sets] for set1 in all_sets])) +1e-5

sns.set(context='paper', style='white', color_codes=True, font_scale=4.5)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
sns.heatmap(data = set_overlap, annot=True, ax=ax, square=True, vmin=.00001, cmap='Oranges', linewidths=.5, cbar=False, norm=LogNorm(), fmt='.2%', xticklabels=ticks, yticklabels=ticks, annot_kws={"size": 'x-small'})
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(triplets_overlap, bbox_inches='tight')
plt.show()

#sources overlap
content_sources_set = set(np.unique(content_triplets.values))
semantic_sources_set = set(np.unique(semantic_triplets.values))
jargon_sources_set = set(np.unique(jargon_triplets.values))
stance_sources_set = set(np.unique(stance_triplets.values))

all_sets = [content_sources_set, semantic_sources_set, jargon_sources_set, stance_sources_set]
ticks = ['copy', 'shift', 'jargon', 'stance']
set_overlap = np.array(([[len(set1.intersection(set2))/len(set1) for set2 in all_sets] for set1 in all_sets]))

sns.set(context='paper', style='white', color_codes=True, font_scale=4.5)
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
sns.heatmap(data = set_overlap, annot=True, ax=ax, square=True, vmin=.00001, cmap='PuBu', linewidths=.5, cbar=False, norm=LogNorm(), fmt='.2%', xticklabels=ticks, yticklabels=ticks, annot_kws={"size": 'x-small'})
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(sources_overlap, bbox_inches='tight')
plt.show()
