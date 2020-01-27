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
from scipy.cluster.hierarchy import ward, dendrogram


MDS()

df = pd.read_excel('Cricket_inc_data.xlsx',sheetname='Test_data')
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

print(totalvocab_stemmed)
print(totalvocab_tokenized)
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

num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print("clusters are: ")
print(clusters)

Y = tfidf_vectorizer.transform(["in porting failed in auto pay"])
prediction = km.predict(Y)
print("Predicted cluster")
print(prediction)
list = list(range(1,num_clusters+1))
print(list)

inc_data = {'incident': inc_list, 'data': clean_data, 'cluster': clusters, 'uts link': inc_links}
frame = pd.DataFrame(inc_data, index = [clusters] , columns = ['incident','data','cluster', 'uts link'])
print(frame['cluster'].value_counts()) #number of INCs per cluster (clusters from 0 to 4)

#grouped = frame['data'].groupby(frame['cluster']) #groupby cluster for aggregation purposes
#average rank (1 to 100) per cluster
#grouped.mean()
print("\n\n")
print("*********   Top terms per cluster:")
print(km.cluster_centers_)
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :3]:  # replace 6 with n words per cluster
        #print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print()
    print("Cluster %d Label count:" % i, end='')
    count=0
    for incident in frame.ix[i]['incident']:
        count = count + 1
        #print('%s,' % incident, end='')
    print(count)
    print()  # add whitespace
    print()
    frame.loc[frame.cluster == i, ['incident', 'uts link']].to_excel('cluster_' + str(i + 1) + ' Cricket_inc_data.xlsx')

print("Creating Graph !!")
exit(0)

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]
print()
print()

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
#set up cluster names using a dict
cluster_names = {0: 'Cluster 1',
                 1: 'Cluster 2',
                 2: 'Cluster 3',
                 3: 'Cluster 4',
                 4: 'Cluster 5'}

# some ipython magic to show the matplotlib plots inline
#%matplotlib inline

# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=inc_list))

## group by cluster
groups = df.groupby('label')
#
## set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
#
## iterate through groups to layer the plot
## note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params( \
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)
plt.show()  # show the plot

# uncomment the below to save the plot if need be
# plt.savefig('clusters_small_noaxes.png', dpi=200)

#linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
#
#fig, ax = plt.subplots(figsize=(15, 20)) # set size
#ax = dendrogram(linkage_matrix, orientation="right", labels=clusters);
#
#plt.tick_params(\
#    axis= 'x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom='off',      # ticks along the bottom edge are off
#    top='off',         # ticks along the top edge are off
#    labelbottom='off')
#
#plt.tight_layout() #show plot with tight layout
#
##uncomment below to save figure
##plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
#plt.close()

print("Finished !!!")