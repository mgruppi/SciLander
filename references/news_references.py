import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from goose3 import Goose
from habanero import counts
from pandarallel import pandarallel
from tqdm import tqdm

from utils import analyze_url

################################# GLOBAL #################################
data_dir = str(Path.home()) + '/data/nela/'

#input
news = pd.read_csv(data_dir + 'input/nela_covid_urls.csv', names=['source', 'url', 'timestamp']).drop_duplicates().reset_index(drop=True)
papers = pd.read_csv(data_dir + 'input/cord19_urls.csv')
blacklist_domains = set(open(data_dir + 'input/blacklist_domains.txt').read().splitlines())
blacklist_references = open(data_dir + 'input/blacklist_references.txt').read().splitlines()
institutions = pd.read_csv(data_dir + 'input/institutions_metadata.tsv', sep='\t')
repositories = pd.read_csv(data_dir + 'input/academic_repositories.csv')
papers['domain'] = papers['url'].apply(lambda x: analyze_url(x)['domain'])
institutions['domain'] = institutions['URL'].apply(lambda x: analyze_url('http://'+x)['domain'])
repositories['domain'] = repositories['URL'].apply(lambda x: analyze_url(x)['domain'])
academic_domains = set(pd.concat([papers['domain'], institutions['domain'], repositories['domain']]).unique())
academic_domains = list(academic_domains.difference(blacklist_domains))

#output
source_sci_references_file = data_dir + 'source_sci_references.csv'
source_other_references_file = data_dir + 'source_other_references.csv'
cord19_altmetric = data_dir + 'citations/cord19_altmetric.csv'
cord19_crossref = data_dir + 'citations/cord19_crossref.csv'

#parallelization
pandarallel.initialize(progress_bar=False, use_memory_fs=False, verbose=1)
tqdm.pandas()

#requests
http_timeout = 3.0
headers = {"User-Agent": "Mozilla/5.0"}
goose = Goose({'keep_footnotes': False})
################################# ###### ################################

################################ HELPERS ################################
#check whether a link is coming from a scientific domain
def is_scientific_link(link):
    link = analyze_url(link)
    return (link['suffix'] in ['.edu', '.gov', '.science']) or (link['domain'] in academic_domains)

#check whether a link is coming from a blacklisted domain
def is_blacklisted_link(link, main_domain):
    l = analyze_url(link)
    return (l['domain'] in blacklist_domains.union(set([main_domain, '']))) or (link.endswith(('.jpg', '.jpeg')))

#find the context of links
def contextualize_links(links, lxml_links):
    contextualize_links = []
    for l in links:
        link = lxml_links[l]
        reference = link.get('href')
        phrase = link.text.replace(u'\xa0', u' ')
        paragraphs = [p for p in re.split('\n+', link.parent.text.replace(u'\xa0', u' '))]
        paragraph = next((p for p in paragraphs if phrase in p), '')
        contextualize_links += [(reference, paragraph, phrase)]
    return contextualize_links

#extract "rel" attribute from links
def get_links_rel(links, lxml_links):
    rel_links = []
    for l in links:
        link = lxml_links[l]
        rel_links += [(link.get('rel'),l)]
    return rel_links

#download and process the html body of each news url 
def _extract_references(news):
    try:
        r = requests.get(news.url, headers=headers, timeout=http_timeout)
        main_domain = analyze_url(r.url)['domain']
        article = BeautifulSoup(r.content.decode('utf-8','ignore'), 'lxml').body
        lxml_links = {link.get('href'):link for link in article.findAll('a')}
        
        links = goose.extract(raw_html=r.content).links
        links = [l for l in links if not is_blacklisted_link(l, main_domain)]
        scientific_links = [l for l in links if is_scientific_link(l)]
        links = list(set(links).difference(set(scientific_links)))

        if scientific_links:
            scientific_links = contextualize_links(scientific_links, lxml_links)
            news['reference'], news['paragraph'], news['phrase'] = zip(*scientific_links)

        if links:
            links = get_links_rel(links, lxml_links)
            news['links'] = links
    except:
        pass
    return news

#send request to CrossRef using doi
def _doi_crossref(doi):
    try: return str(counts.citation_count(doi = doi))
    except: return ''

#send request to Altmetric using doi
def _doi_altmetric(doi, KEY):
    try: return requests.get('https://api.altmetric.com/v1/doi/'+doi+KEY, timeout=http_timeout).text
    except: return ''
################################ ####### ################################

#extract references from news
def extract_references(news):
    news = news.parallel_apply(_extract_references, axis=1)
    news[['source', 'url', 'links']].to_csv(source_other_references_file, index=False)
    news = news[['source', 'url', 'reference', 'phrase', 'paragraph']]
    news = news.dropna().set_index(['source', 'url']).apply(pd.Series.explode).reset_index()
    news.to_csv(source_sci_references_file, index=False)

#request auxiliary data from Altmetric
def request_altmetric(papers):
    KEY = '?key='+ os.getenv('ALTMETRIC_KEY')
    papers = papers.drop_duplicates().dropna()
    papers['altmetric_info'] = papers.progress_apply(lambda p: _doi_altmetric(p['doi']), axis=1)
    papers = papers[~papers.altmetric_info.str.contains('Not Found')].dropna()
    papers[['readers_count', 'posts_count']] = papers.apply(lambda p: (lambda j: (j['readers_count'], j['cited_by_posts_count']))(json.loads(p['altmetric_info']), KEY), axis=1, result_type='expand')
    papers = papers[['url', 'readers_count', 'posts_count']]
    papers.to_csv(cord19_altmetric, index=False)

#request auxiliary data from CrossRef
def request_crossref(papers):
    papers = papers.drop_duplicates().dropna()
    papers['crossref_citations'] = papers.progress_apply(lambda p: _doi_crossref(p['doi']), axis=1)
    papers.to_csv(cord19_crossref, index=False)

if __name__ == "__main__":
    extract_references(news)
    # request_altmetric(papers)
    # request_crossref(papers)    
