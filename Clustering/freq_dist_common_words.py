import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import codecs

df = pd.read_excel('Airtel.xlsx',sheetname='Data')
data = ""
for i in df.index:
    data += df['Notes'][i]

# NLTK's default stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))

# We're adding some on our own - could be done inline like this...
# custom_stopwords = set((u'–', u'dass', u'mehr'))
# ... but let's read them from a file instead (one stopword per line, UTF-8)
#stopwords_file = './stopwords.txt'
custom_stopwords = set(codecs.open('stopwords_file', 'r', 'utf-8').read().splitlines())
all_stopwords = default_stopwords | custom_stopwords
print(all_stopwords)
words = nltk.word_tokenize(data)

# Remove single-character tokens (mostly punctuation)
words = [word for word in words if len(word) > 1]

# Remove numbers
words = [word for word in words if not word.isnumeric()]

# Lowercase all words (default_stopwords are lowercase too)
words = [word.lower() for word in words]

# Stemming words seems to make matters worse, disabled
# stemmer = nltk.stem.snowball.SnowballStemmer('german')
# words = [stemmer.stem(word) for word in words]

# Remove stopwords
words = [word for word in words if word not in all_stopwords]

print(words)

# Calculate frequency distribution
fdist = nltk.FreqDist(words)

# Output top 50 words
#for word, frequency in fdist.most_common(50):
#    print(u'{};{}'.format(word, frequency))


print("reading finished")

#'';58969
#*************************;58160
#number;49714
#order;47627
#pending;39293
#team;34792
#please;31437
#activation;30067
#kindly;26948
#id;23405
#dear;23170
#customer;22805
#***************************;22540
#issue;17366
#help;15367
#gt;14445
#check;13692
#showing;13227
#account;13006
#mobile;12971
#sr;12824
#error;12207
#icrm;11797
#unable;11327
#hi;10911
#name;9944
#clear;9798
#sim;9735
#change;9704
#task;9297
#regards;8654
#status;8600
#new;8557
#still;8498
#postpaid;8010
#generated;7799
#--;7588
#request;7385
#plan;7156
#resolve;7156
#close;6825
#airtel;6810
#details;5925
#service;5865
#pls;5776
#circle;5747
#mentioned;5662
#attached;5603
#need;5588
#``;5578
