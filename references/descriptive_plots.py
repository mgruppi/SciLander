
import re
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from textstat.textstat import textstat

from utils import analyze_url, clean_reference, preprocess_input

################################# GLOBAL ################################
data_dir = str(Path.home()) + '/data/nela/'

source_sci_references_file = data_dir + 'source_sci_references.csv'
source_other_references_file = data_dir + 'source_other_references.csv'
top_references = data_dir + 'plots/top_references.png'
references_per_article_kde = data_dir + 'plots/references_per_article_kde.png'
references_per_article_top = data_dir + 'plots/references_per_article_top.png'
context_stats = lambda x: data_dir + 'plots/'+x+'.png'
cord19_popularity = lambda x: data_dir + 'plots/cord19_popularity_'+x+'.png'
stance_per_month = data_dir + 'plots/stance_per_month.png'
unreliable_whitelist_file = data_dir + 'input/unreliable_whitelist.txt'
unreliable_references_file = data_dir + 'unreliable_references.csv'
references_delta = data_dir + 'plots/references_delta.png'
term_related_references_proportion = lambda term: data_dir + 'plots/'+term+'_related_references_proportion.png'
term_related_references_stance = lambda term: data_dir + 'plots/'+term+'_related_references_stance.png'

cord19_altmetric = data_dir + 'citations/cord19_altmetric.csv'
cord19_crossref = data_dir + 'citations/cord19_crossref.csv'
cord19_full = data_dir + 'input/cord19_urls.csv'
################################# ###### ################################


#plot top references with horizontal bars
def plot_top_references(top=20):
    #prepare data
    references = pd.read_csv(source_sci_references_file).rename(columns={'reference': 'scientific domain'})
    references['scientific domain'] = references['scientific domain'].apply(lambda x: (lambda y: y['domain']+'.'+y['suffix'])(analyze_url(x)))
    data = references[['scientific domain', 'label']].value_counts().reset_index().rename(columns={0: 'count'})

    #fill missing values
    all_refs = data[['scientific domain']].drop_duplicates()
    all_refs['label_0'], all_refs['label_1'] = '0', '1'
    all_refs = pd.concat([all_refs[['scientific domain', 'label_0']].rename(columns={'label_0': 'label'}), all_refs[['scientific domain', 'label_1']].rename(columns={'label_1': 'label'})])
    data = data.merge(all_refs, on=['scientific domain', 'label'], how='outer').fillna(0)

    #normalize
    data = data.merge(data.groupby('scientific domain')['count'].sum(), on='scientific domain')
    data['percentage of references'] = data['count_x']/data['count_y'] * 100
    data['label'] = data['label'].map({'0': 'reliable', '1': 'unreliable'})

    #select top refs
    selected_refs = data[['scientific domain', 'count_y']].drop_duplicates().sort_values(by='count_y', ascending=False)['scientific domain'][:top]
    data = data.set_index('scientific domain').loc[selected_refs].drop(['count_x', 'count_y'], axis=1).reset_index()
    data.loc[data['label']=='unreliable', 'percentage of references'] = 100

    #plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    sns.barplot(x='percentage of references', y='scientific domain', color='#ef8a62', data=data[data['label']=='unreliable'])
    sns.barplot(x='percentage of references', y='scientific domain', color='#67a9cf', data=data[data['label']=='reliable'])
    top_bar = mpatches.Patch(color='#67a9cf', label='reliable')
    bottom_bar = mpatches.Patch(color='#ef8a62', label='unreliable')
    plt.legend(handles=[top_bar, bottom_bar], loc = 'lower right', bbox_to_anchor = [.96, 0], title='label')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(top_references, bbox_inches='tight')
    plt.show()

#plot distribution for reliable and unreliable sources
def plot_references_per_article(top=5):
    #prepare data
    references = pd.read_csv(source_sci_references_file)
    news,_,_,_,_ = preprocess_input()
    news = news.groupby(['source', 'label'])['url'].count().reset_index('label').rename(columns={'url':'#articles'})
    references = references.groupby('source')['reference'].count().apply(pd.Series).rename(columns={0:'#references'})
    data = news.join(references).fillna(0)
    data['references/article'] = data['#references'] /  data['#articles']
    data['label'] = data['label'].map({0: 'reliable', 1: 'unreliable'})

    #visualize using KDE
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    sns.kdeplot(data=data, x='references/article', hue='label', cut=0, fill=True, palette='RdBu')
    ax.set_xscale('log')
    ax.legend(['reliable', 'unreliable'], title='label')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(references_per_article_kde, bbox_inches='tight')
    plt.show()

    #focus on top sources w.r.t. references/article
    data_reliable = data[data['label']=='reliable'].sort_values(by='references/article')[-top:].reset_index()
    data_unreliable = data[data['label']=='unreliable'].sort_values(by='references/article')[-top:].reset_index()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,10), sharex=True)
    left_bar = mpatches.Patch(color='#ef8a62', label='unreliable')
    right_bar = mpatches.Patch(color='#67a9cf', label='reliable')
    fig.legend(handles=[right_bar, left_bar], loc = 'upper right', title='label')

    sns.barplot(data=data_unreliable, x='references/article', y='source', color='#ef8a62', ax=ax[0])
    sns.barplot(data=data_reliable, x='references/article', y='source', color='#67a9cf', ax=ax[1])
    ax[0].set_xlabel('')
    [ax[i].set_ylabel('') for i in range(2)]
    fig.supylabel('source', fontsize='medium')
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(references_per_article_top, bbox_inches='tight')
    plt.show()

#plot context length (length of paragraphs with references) for reliable and unreliable sources
def plot_context_stats(stat, top=5):
    #prepare data
    references = pd.read_csv(source_sci_references_file)
    if stat == 'length':
        xaxis = 'Context Length (words)'
        references[xaxis] = references['paragraph'].str.findall(r'(\w+)').str.len()
    if stat == 'readability':
        xaxis = 'Context Readability'
        references[xaxis] = references['paragraph'].apply(lambda x: textstat.flesch_reading_ease(str(x)))

    data = references.groupby(['source', 'label'])[xaxis].median().reset_index()
    data['label'] = data['label'].map({'0': 'reliable', '1': 'unreliable'})
    
    #visualize using KDE
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    sns.kdeplot(data=data, x=xaxis, hue='label', fill=True, palette='RdBu')
    ax.set_xscale('log')
    ax.legend(['reliable', 'unreliable'], title='label')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(context_stats(xaxis.lower().replace(' ', '_')+'_kde'), bbox_inches='tight')
    plt.show()

    #focus on bottom sources
    data_reliable = data[data['label']=='reliable'].sort_values(by=xaxis)[:top].reset_index()
    data_unreliable = data[data['label']=='unreliable'].sort_values(by=xaxis)[:top].reset_index()

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20,10), sharex=True)
    left_bar = mpatches.Patch(color='#ef8a62', label='unreliable')
    right_bar = mpatches.Patch(color='#67a9cf', label='reliable')
    fig.legend(handles=[right_bar, left_bar], loc = 'upper right', title='label')

    sns.barplot(data=data_unreliable, x=xaxis, y='source', color='#ef8a62', ax=ax[0])
    sns.barplot(data=data_reliable, x=xaxis, y='source', color='#67a9cf', ax=ax[1])
    ax[0].set_xlabel('')
    [ax[i].set_ylabel('') for i in range(2)]
    fig.supylabel('source', fontsize='medium')
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(context_stats(xaxis.lower().replace(' ', '_')+'_bottom'), bbox_inches='tight')
    plt.show()

#plot auxiliary metrics for cord19 references
def plot_cord19_popularity():
    #prepare data
    references = pd.read_csv(source_sci_references_file)
    cord19 = pd.read_csv(cord19_crossref).merge(pd.read_csv(cord19_altmetric))
    cord19['url'] = cord19['url'].apply(lambda u: re.sub(';.*', '', u))
    cord19['clean_url'] = clean_reference(cord19['url'])
    references['clean_url'] = clean_reference(references['reference'])
    data = references[['clean_url', 'label']].merge(cord19, on='clean_url')[['url', 'crossref_citations', 'readers_count', 'posts_count', 'label']]
    data['label'] = data['label'].map({'0':'reliable', '1':'unreliable'})
    data['scientific domain'] = data['url'].apply(lambda x: (lambda y: y['domain']+'.'+y['suffix'])(analyze_url(x)))
    data = data[data['scientific domain'].isin(['nih.gov', 'sciencedirect.com', 'doi.org'])]
    data = data.rename(columns={'crossref_citations': '#citations', 'readers_count': '#readers', 'posts_count': '#social media posts'})
    data['#citations']=data['#citations'] + 1
    data['#social media posts']=data['#social media posts'] + 1
    
    #scatterplot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    g = sns.jointplot(data=data, x='#citations', y='#social media posts', hue='label', palette='RdBu')    
    g.plot_joint(sns.scatterplot)
    g.plot_marginals(sns.kdeplot)
    g.ax_joint.set_xscale('log')
    g.ax_joint.set_yscale('log')
    g.ax_joint.set_xticks([1e+0, 1e+1,1e+2,1e+3,1e+4])
    g.ax_joint.set_yticks([1e+0, 1e+1,1e+2,1e+3,1e+4])
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend([], [], frameon=False)
    g.fig.legend([handles[1], handles[0]], [labels[1], labels[0]], title='label', loc='upper center', bbox_to_anchor = [0.5, 1.15], ncol=2)
    plt.gca().set_xticks([])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(cord19_popularity('scatterplot'), bbox_inches='tight')
    plt.show()

    #boxplot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10), sharey=True)
    for i, metric in enumerate(['#citations', '#social media posts']):
        sns.boxplot(x=metric, y='scientific domain', hue='label', data=data, whis=[0, 100], palette="RdBu", ax=ax[i])
        ax[i].legend([],[], frameon=False)
        ax[i].set_xscale('log')
        ax[i].set_ylabel('')
    ax[0].set_ylabel('scientific domain')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend([handles[1], handles[0]], [labels[1], labels[0]], title='label', loc='upper right', bbox_to_anchor = [1.03, 1])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(cord19_popularity('boxplot'), bbox_inches='tight')
    plt.show()

    #print ratios
    news_ratio = data['url'].drop_duplicates().count()/references['clean_url'].drop_duplicates().count()
    cord19_ratio = data['url'].drop_duplicates().count()/cord19['clean_url'].drop_duplicates().count()
    print('News References in CORD19:','{:.1%}'.format(news_ratio))
    print('CORD19 papers in News:','{:.1%}'.format(cord19_ratio))

#plot stance towards science per month
def plot_stance_per_month():
    #prepare data
    source_references = pd.read_csv(source_sci_references_file).dropna()
    source_references['clean_reference'] = clean_reference(source_references['reference'])
    source_references['month'] = source_references['published_utc'].apply(lambda t: datetime.fromtimestamp(t).strftime('%m/%Y')).astype('period[M]')
    source_references['stance'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(source_references[['stance']]).T[0]

    #compute prevalence
    source_counts = source_references[['source', 'label']].drop_duplicates()['label'].value_counts()
    prevalence = source_references.groupby(['label', 'clean_reference'])['source'].nunique().rename('prevalence').reset_index()
    prevalence['prevalence'] = prevalence.apply(lambda p: p['prevalence']/source_counts[p['label']], axis = 1)

    #merge prevalence with data
    data = source_references.groupby(['month', 'clean_reference', 'label', 'source'], as_index=False)['stance'].mean().groupby(['month', 'clean_reference', 'label'], as_index=False)['stance'].median()
    data = data.merge(prevalence, on=['label', 'clean_reference']).sort_values(by='month')
    data['month'] = data['month'].apply(lambda m: m.strftime("%B"))

    #compute normalized stance
    data['stance'] = data['stance'] * data['prevalence']
    data['stance'] = StandardScaler().fit_transform(data[['stance']]).T[0]

    #plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    sns.lineplot(data=data, x='month', y='stance', hue='label', hue_order=[1, 0], palette="RdBu")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend([handles[1], handles[0]], ['reliable', 'unreliable'], title='label', loc='upper right')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(stance_per_month, bbox_inches='tight')
    plt.show()

#plot popular references of unreliable sources
def plot_unreliable_references(count_limit=100):
    unreliable_whitelist = set(open(unreliable_whitelist_file).read().splitlines())
    references = pd.read_csv(source_other_references_file)
    #keep only unreliable sources
    references = references[references['label'] == 1]['links'].dropna()
    references = references.apply(eval).explode().apply(pd.Series).rename(columns={0:'rel', 1:'url'})
    references['domain'] = references['url'].apply(lambda x: (lambda y: y['domain']+'.'+y['suffix'])(analyze_url(x)))
    #remove short urls
    references = references[references['url'].apply(lambda x: (lambda y: y['path'] not in ['/', ''])(analyze_url(x)))]
    #keep only references from whitelist
    references = references[references['domain'].str.match('('+'|'.join(unreliable_whitelist)+')')]
    #dump results
    references['domain'].value_counts()[references['domain'].value_counts() > count_limit].to_csv(unreliable_references_file)

#plot the time delta between the first reference from a reliable source and the first reference from an unreliable source
def plot_references_delta():
    references = pd.read_csv(source_sci_references_file)
    references['clean_url'] = clean_reference(references['reference'])
    data = references[['clean_url', 'label', 'published_utc']].dropna(subset=['published_utc'])
    data['label'] = data['label'].map({'0':'reliable', '1':'unreliable'})
    data['published_utc'] = data['published_utc'].apply(lambda t: datetime.fromtimestamp(t))

    def get_delta_in_hours(reference):
        mins = reference.groupby('label')['published_utc'].min()
        delta = int((mins['reliable'] - mins['unreliable']).total_seconds()/3600) if all([k in mins.keys() for k in ['reliable', 'unreliable']]) else 9000 if 'reliable' not in mins.keys() else -9000
        return delta

    data = data.groupby(['clean_url'])['label', 'published_utc'].apply(get_delta_in_hours)
    data = data.dropna().reset_index().rename(columns={0: 'delta'})

    #plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    sns.histplot(data=data, x='delta', palette="RdBu", ax=ax)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(references_delta, bbox_inches='tight')
    plt.show()

#plot temporal distribution and stance of references related to a given term
def plot_term_related_references(term):
    references = pd.read_csv(source_sci_references_file).dropna(subset=['published_utc', 'paragraph', 'stance'])
    references['label'] = references['label'].map({'0':'reliable', 1:'unreliable'})
    references['published_utc'] = references['published_utc'].apply(lambda t: datetime.fromtimestamp(t)).dt.strftime('%Y-%m')
    references['published_utc'] = pd.to_datetime(references['published_utc'], format='%Y-%m')
    references['clean_url'] = clean_reference(references['reference'])

    #find term-related references
    term_refs = references[references['paragraph'].str.contains(term, case=False)]['clean_url'].drop_duplicates()

    #article proportion
    data = references[references['clean_url'].isin(term_refs)]
    data = pd.DataFrame(references.groupby(['published_utc', 'label', 'source']).size(), columns=['total counts']).join(pd.DataFrame(data.groupby(['published_utc', 'label', 'source']).size(), columns=['counts'])).fillna(0)

    data['Article Proportion'] = data['counts'] / data['total counts']
    data = data.reset_index()
    data['published_utc'] = data['published_utc'].dt.strftime('%Y-%m')
    data = data[data['published_utc'] != '2021-01']
    data = data.rename(columns={'published_utc': 'Publication Month', 'label': 'Label'})

    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    sns.barplot(data=data, x='Publication Month', y='Article Proportion', hue='Label', hue_order=['unreliable', 'reliable'], palette="RdBu", ax=ax)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(term_related_references_proportion(term), bbox_inches='tight')
    plt.show()

    #stance
    data = references[references['clean_url'].isin(term_refs)]
    data = data.groupby(['published_utc', 'label', 'source'])['stance'].median().reset_index()
    data['published_utc'] = data['published_utc'].dt.strftime('%Y-%m')
    data = data[data['published_utc'] != '2021-01']
    data = data.rename(columns={'published_utc': 'Publication Month', 'stance': 'Stance', 'label': 'Label'})

    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    sns.boxplot(data=data, x='Publication Month', y='Stance', hue='Label', hue_order=['unreliable', 'reliable'], palette="RdBu", ax=ax)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(term_related_references_stance(term), bbox_inches='tight')
    plt.show()

    #top-5 refs
    print(references[(references['clean_url'].isin(term_refs)) & (references['label'] == 'unreliable')]['reference'].value_counts()[:5])


if __name__ == "__main__":
    # plot_top_references()
    # plot_references_per_article()
    # plot_context_stats('length')
    # plot_context_stats('readability')
    # plot_cord19_popularity()
    # plot_stance_per_month()
    # plot_unreliable_references()
    # plot_references_delta()
    plot_term_related_references('hydroxychloroquine|HCQ')
