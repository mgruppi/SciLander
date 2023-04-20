import re
import string
from pathlib import Path
from urllib.parse import urlsplit

import requests
import tldextract
from bs4 import BeautifulSoup
import pandas as pd

################################# GLOBAL ################################
data_dir = str(Path.home()) + '/data/nela/'
http_timeout = 3
################################# ###### ################################

#Find the domain, the path, and the query of a url
def analyze_url(url):
    try:
        assert(url.startswith('http'))
        tld = tldextract.extract(url)
        domain = tld.domain
        suffix = tld.suffix
        url = urlsplit(url)
        path = url.path
        query = url.query
    except:
        domain, path, query, suffix = '', '', '', ''

    return {'domain': domain, 'path': path, 'query': query, 'suffix': suffix}

#clean url of reference
def clean_reference(reference):
    return reference.apply(lambda x: (lambda y: y['domain']+'.'+y['suffix']+y['path'])(analyze_url(x)))\
                     .apply(lambda x: re.sub(r'[^\S]+', '', x).encode("ascii", errors="ignore").decode())\
                     .apply(lambda x: re.sub(r'((v[1-9])?\.(pdf|full\.pdf|htm[l]?)|(v[1-9])?|\/fulltext|#.*)$', '', x))

#Scrap CDC diseases and conditions
def scrap_CDC_vocabulary():
    vocabulary = []
    for az in list(string.ascii_lowercase) + ['0']:
        r = requests.get('https://www.cdc.gov/diseasesconditions/az/'+az+'.html', timeout=http_timeout)
        div = BeautifulSoup(r.content.decode('utf-8','ignore'), 'lxml').find('div', attrs={'class' : 'az-content col'})
        vocabulary += [a.text for a in div.find_all('a')]

    #clean
    vocabulary = [w for v in vocabulary for w in v.split(' — see ')]
    vocabulary = [w.replace(']', '') for v in vocabulary for w in v.split(' [')]
    vocabulary = [w.replace(')', '') for v in vocabulary for w in v.split(' (')]
    vocabulary = [w.replace('see also ', '') for w in vocabulary]

    #dedup
    vocabulary = [v for i, v in enumerate(vocabulary) if v not in vocabulary[:i]]

    #store
    with open(data_dir + 'CDC_vocab.txt', 'w') as f:
        [f.write(v + '\n') for v in vocabulary]
