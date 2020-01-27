import pandas
import os
import nltk
from nltk.corpus.reader import CategorizedTaggedCorpusReader

corpus_root = 'C:/ProgramData/Anaconda3/pkgs/nltk_data/corpora/brown'
#newcorpus = TaggedCorpusReader ('./', '.*')
#print(newcorpus.fileids())
#print(newcorpus.words())
reader = CategorizedTaggedCorpusReader("./", r"^[^.]*$", cat_file="cats.txt")
#print(reader.fileids())
#print(reader.categories())
print(reader.tagged_words())

suffix_fdist = nltk.FreqDist()
for word in reader.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1

print(suffix_fdist)

common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features

tagged_words = reader.tagged_words(categories='cat1')
print(tagged_words)
featuresets = [(pos_features(n), g) for (n, g) in tagged_words]
print(featuresets)

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.DecisionTreeClassifier.train(train_set)
#print(classifier.classify(pos_features('The Fulton County Grand Jury said Friday an investigation of Atlanta''s recent primary election produced no evidence that any irregularities took place.')))
print(nltk.classify.accuracy(classifier, test_set))
print(classifier.classify(pos_features('cancelled')))
