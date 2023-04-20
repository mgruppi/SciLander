import os
import pickle
import re
from math import sqrt

import numpy as np
import pandas as pd
import spacy
from empath import Empath
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

############################### CONSTANTS ###############################

#LDA parameters
numOfTopics = 16
max_iter = 100

#Minimum length for paragraphs (#chars)
MIN_PAR_LENGTH = 256

#Weights for similarities
vecW = .5
entW = .25
topW = .25

############################### ######### ###############################

################################ MODELS #################################
nlp = spacy.load('en_core_web_lg')

lexicon = Empath()
scilexicon = list(set([w for w in lexicon.cats["health"]+lexicon.cats["science"] if '_' not in w]))

def topic_model():

    if os.path.isfile('models/topic_model.lda') and os.path.isfile('models/topic_model.vec'):
        lda = pickle.load(open('models/topic_model.lda', 'rb'))
        tf_vectorizer = pickle.load(open('models/topic_model.vec', 'rb'))
        return lda, tf_vectorizer

    article = pd.read_csv("sample_data/article.tsv", sep='\t')
    papers = pd.read_csv("sample_data/papers.tsv", sep='\t')
    df = pd.concat([article, papers], ignore_index=True)
    
    #define vectorizer
    tf_vectorizer = CountVectorizer(stop_words='english', vocabulary=scilexicon)
    tf = tf_vectorizer.transform(df['full_text'])

    #fit lda topic model
    print('Fitting LDA model...')
    lda = LatentDirichletAllocation(n_components=numOfTopics, max_iter=max_iter, learning_method='online', n_jobs=-1)
    lda.fit(tf)

    #cache model
    os.makedirs('models', exist_ok=True)
    pickle.dump(lda, open('models/topic_model.lda', 'wb'))
    pickle.dump(tf_vectorizer, open('models/topic_model.vec', 'wb'))


    return lda, tf_vectorizer


lda, tf_vectorizer = topic_model()

################################ ###### #################################

################################ EXPAND TEXT ################################

#Split text to paragraph
def text2par(text):
    return [p for p in re.split('\n', text) if len(p) > MIN_PAR_LENGTH]

#Extract entities from text
def text2ent(text):
    return set([e.text for e in nlp(text).ents if e.label_ in ['PERSON', 'ORG', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'CARDINAL']])

#Compute text vector
def text2vec(text):
    return nlp(text).vector

#Compute text topics
def text2top(text):
    return lda.transform(tf_vectorizer.transform([text])).tolist()[0]


#Extract text vector, entities, topics
def expand_text(text):
    text_vec = text2vec(text)
    text_ent = text2ent(text)
    text_top = text2top(text)
    return text_vec, text_ent, text_top

################################ ########### ################################


############################### TEXT SIMILARITY ###############################

#Compute cosine similarity between two vectors
def vec_sim(a_par_vec, p_par_vec):
    return abs(cosine_similarity(a_par_vec.reshape(1, -1), p_par_vec.reshape(1, -1))[0][0])

#Compute jaccard similarity between two sets of entities
def ent_sim(a_par_ent, p_par_ent):
    return len(a_par_ent.intersection(p_par_ent)) / (len(a_par_ent.union(p_par_ent)) + 0.1)

#Compute Hellinger similarity between two topic distributions
def top_sim(a_par_top, p_par_top): 
    sim = 0.0
    #if both are not related to any topic return 0
    if not (len(set(a_par_top)) == len(set(p_par_top)) == 1 and set(a_par_top).pop() == set(p_par_top).pop() == 0.0625):    
        for tx, ty in zip(a_par_top, p_par_top):
            sim += (sqrt(tx) - sqrt(ty))**2
        sim = 1 - 1/sqrt(2) * sqrt(sim)
    return sim

#Compute semantic textual similarity
def semantic_textual_similarity(a_par_vec, a_par_ent, a_par_top, p_par_vec, p_par_ent, p_par_top):
    sim = 0.0    
    sim += vecW * vec_sim(a_par_vec, p_par_vec)
    sim += entW * ent_sim(a_par_ent, p_par_ent)
    sim += topW * top_sim(a_par_top, p_par_top)
    return sim

############################### ############### ###############################

def scientific_coherence(article, papers):

    papers['full_text'] = papers['full_text'].apply(text2par)
    papers = papers.explode('full_text')
    papers = papers.set_index('full_text')
    papers_paragraphs = list(papers.index)

    for i in range(len(article)):
        article_text = article['full_text'].iloc[i]
        article_paragraphs = text2par(article_text) 

        all_sims = []
        for a_par in article_paragraphs:
            max_sim = 0
            a_par_vec, a_par_ent, a_par_top = expand_text(a_par)
            for p_par in papers_paragraphs:
                p_par_vec, p_par_ent, p_par_top = expand_text(p_par)
                sim = semantic_textual_similarity(a_par_vec, a_par_ent, a_par_top, p_par_vec, p_par_ent, p_par_top)
                if sim > max_sim:
                    max_sim = sim
            #print(a_par, p_par)
            all_sims += [max_sim]

        print('Scientific Coherence of %s is %f'%(article['url'].iloc[i], np.mean(all_sims)))


if __name__ == "__main__":
    article = pd.read_csv("sample_data/article.tsv", sep='\t')
    papers = pd.read_csv("sample_data/papers.tsv", sep='\t')
    scientific_coherence(article, papers)
