from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score


MDS()

df = pd.read_excel('Cricket_inc_data_orig.xlsx',sheetname='Test_data')
data=[]
inc_list=[]
inc_links=[]
for i in df.index:
    data.append(df['Inc Summary'][i])
    inc_list.append(df['Inc ID'][i])
    inc_links.append(df['Inc Uts Link'][i])

print("reading finished")

stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.add("please")
stopwords.add("Please")
stopwords.add("kindly")
stopwords.add("Kindly")
stopwords.add("PLEASE")
stopwords.add("KINDLY")
stopwords.add("Airtel")
stopwords.add("AIRTEL")
stopwords.add("team")
stopwords.add("Team")
stopwords.add("TEAM")
stopwords.add("dear")
stopwords.add("Dear")
stopwords.add("DEAR")

stemmer = SnowballStemmer("english")

## here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
def tokenize_and_stem(text):
     # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        #token = re.sub('[!@#$-_]', ' ', token)
        token = token.translate({ord(c): None for c in '!@#$-_'})
        if re.search('[a-zA-Z]', token):
            if re.search('[0-9]', token):
                token = ''.join([i for i in token if not i.isdigit()])
                filtered_tokens.append(token)
            else:
                filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        #token = re.sub('[!@#$-_]', ' ', token)
        token = token.translate({ord(c): None for c in '!@#$-_'})
        if re.search('[a-zA-Z]', token):
            if re.search('[0-9]', token):
                token = ''.join([i for i in token if not i.isdigit()])
                filtered_tokens.append(token)
            else:
                filtered_tokens.append(token)
    return filtered_tokens

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    temp = [w for w in tokens if w.lower() not in stopwords]
    temp_1 = ' '.join(temp)
    return temp_1


# not super pythonic, no, not at all.
# use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
print("creating vocab")
clean_data = []
for i in data:
    i = remove_stopwords(i)
    clean_data.append(i)
    allwords_stemmed = tokenize_and_stem(i)  # for each item in data, tokenize/stem
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

#print(totalvocab_stemmed)
#print(totalvocab_tokenized)
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

#define vectorizer parameters
print("creating vecorizer")
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.1, stop_words=stopwords,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_data)
print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

dist = 1 - cosine_similarity(tfidf_matrix)
range_n_clusters = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

for num_clusters in range_n_clusters:
	km = KMeans(n_clusters=num_clusters, random_state=10)
	#km.fit(tfidf_matrix)
	#clusters = km.labels_.tolist()
	clusters = km.fit_predict(tfidf_matrix)
	print(clusters)
	silhouette_avg = silhouette_score(tfidf_matrix, clusters)
	print("For n_clusters =", num_clusters,
          "The average silhouette_score is :", silhouette_avg)


#inc_data = {'incident': inc_list, 'data': clean_data, 'cluster': clusters, 'uts link': inc_links}
#frame = pd.DataFrame(inc_data, index = [clusters] , columns = ['incident','data','cluster', 'uts link'])
#print(frame['cluster'].value_counts()) #number of INCs per cluster (clusters from 0 to 4)

print("Finished !!!")