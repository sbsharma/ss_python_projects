
import nltk
import random

corpus_root = './Categorization_Word_token/'

from nltk.corpus.reader import CategorizedTaggedCorpusReader

reader = CategorizedTaggedCorpusReader( corpus_root, r'.*', cat_file = "cats.txt" )

documents = [(list(reader.words(fileid)), category)
            for category in reader.categories()
            for fileid in reader.fileids(category)]
random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in reader.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]
classifier = nltk.NaiveBayesClassifier.train(featuresets)

#classifier.show_most_informative_features(15)
# To Test Classifier
print(nltk.classify.accuracy(classifier, featuresets))
print(classifier.classify(document_features('Description for ticket to categorizes!')))
