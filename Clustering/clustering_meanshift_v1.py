from __future__ import print_function

import re
from itertools import cycle

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_excel('Airtel.xlsx',sheetname='Data')
data=[]
inc_list=[]
for i in df.index:
    data.append(df['Notes'][i])
    inc_list.append(df['Incident ID'][i])

print("reading finished for ")
print(df.index)

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
stopwords.add("order")
stopwords.add("Order")
stopwords.add("ID")
stopwords.add("Id")
stopwords.add("id")

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

print(totalvocab_stemmed)
print(totalvocab_tokenized)

vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words=stopwords,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = vectorizer.fit_transform(clean_data)
print(tfidf_matrix.shape)
dense_text = tfidf_matrix.todense()


# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(dense_text, quantile=0.2, n_samples=1000)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(dense_text)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(dense_text[my_members, 0], dense_text[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()



