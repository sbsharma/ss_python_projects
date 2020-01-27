# Check the versions of libraries
# Python version
import sys
#print('Python: {}'.format(sys.version))
# scipy
import scipy
#print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
#print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
#print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
#print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
#print('sklearn: {}'.format(sklearn.__version__))
import nltk
print ('nltk: {}'.format(nltk.__version__))
# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from nltk.corpus.reader import CategorizedTaggedCorpusReader
import os

# Load dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = pandas.read_csv (url, names=names)
#print ('dataset loaded')
#print (dataset.shape)
#print (dataset.describe())
#print (dataset.groupby('class').size())
#dataset.plot(kind='box')
#plt.show()
#
##scatter plot matrix
#scatter_matrix(dataset)
#plt.show()

#
#models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
## Split-out validation datase
#array = dataset.values
#X = array[:,0:4]
#Y = array[:,4]
#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#
## evaluate each model in turn
#results = []
#names = []
#scoring = 'accuracy'
#for name, model in models:
#	kfold = model_selection.KFold(n_splits=10, random_state=seed)
#	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#	results.append(cv_results)
#	names.append(name)
#	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#	print(msg)
#

#df = pandas.read_excel("test_category.xlsx", sheetname="Nov 2017")
#print (df.head())
#print (df['Category 1'])
#for i in df.index:
#   print(df['Description'][i])

#print ('CL1')
#print (df.groupby('Case Level 1').size())
#print ('CL2')
#print (df.groupby('Case Level 2').size())
#print ('CL3')
#print (df.groupby('Case Level 3').size())
#print ('CL4')
#print (df.groupby('Case Level 4').size())
#print (df.groupby('Description').size())

#def ie_preprocess(document):
#    sentences = nltk.sent_tokenize(document)
#    sentences = [nltk.word_tokenize(sent) for sent in sentences]
#    sentences = [nltk.pos_tag(sent) for sent in sentences]
#    print(sentences)#
#
#ie_preprocess("The fourth Wells account moving to another agency is the packaged paper-products division of Georgia-Pacific Corp., which arrived at Wells only last fall. Like Hertz and the History Channel, it is also leaving for an Omnicom-owned agency, the BBDO South unit of BBDO Worldwide. BBDO South in Atlanta, which handles corporate advertising for Georgia-Pacific, will assume additional duties for brands like Angel Soft toilet tissue and Sparkle paper towels, said Ken Haldin, a spokesman for Georgia-Pacific in Atlanta.")

#text = nltk.word_tokenize("The fourth Wells account moving to another agency is the packaged paper-products division of Georgia-Pacific Corp., which arrived at Wells only last fall. Like Hertz and the History Channel, it is also leaving for an Omnicom-owned agency, the BBDO South unit of BBDO Worldwide. BBDO South in Atlanta, which handles corporate advertising for Georgia-Pacific, will assume additional duties for brands like Angel Soft toilet tissue and Sparkle paper towels, said Ken Haldin, a spokesman for Georgia-Pacific in Atlanta.")
#tagged = nltk.pos_tag(text)
#print(tagged)
#str = ' '.join([nltk.tag.tuple2str(tup) for tup in tagged])
#print (str)

df = pandas.read_excel('test_category.xlsx',sheetname='Nov 2017')
# To Create Corpus Files
cats = open("cats.txt", "wt+")
filename = "c"
dict = {}
for i in df.index:
    path = './'
    os.chdir(path)
# Check current working directory.
#    retval = os.getcwd()
#    print('Directory changed successfully %s" % retval')
    #filename = '%s'%df['ETM CT#'][i]
    #print(filename)
    f = open(filename, 'w')
    content_text = nltk.word_tokenize("%s"% df['Description'][i])
    content_text = nltk.pos_tag(content_text)
    content_text = ' '.join([nltk.tag.tuple2str(tup) for tup in content_text])
    #print(content_text)
    f.write('%s' % content_text)
    f.close()
    #print (filename+' '+str(df['Category 1'][i])+'\n')
    cats.write(filename+' '+str(df['Category 1'][i])+'\n')

cats.close()

#corpus_root = 'C:/Users/sausharm/AppData/Roaming/nltk_data/corpora/brown'
#newcorpus = TaggedCorpusReader (corpus_root, '.*')
#print(newcorpus.fileids())
#print(newcorpus.words())
#reader = CategorizedTaggedCorpusReader("./", r"^[^.]*$", cat_file="cats.txt")
#print(reader.categories())


#grammar = r"""
#  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
#      {<NNP>+}                # chunk sequences of proper nouns
#"""
#cp = nltk.RegexpParser(grammar)
#sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"),
#                 ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]
#result = cp.parse(sentence)
#result.draw()
#cp = nltk.RegexpParser('CHUNK: {<V.*> <to> <V.*>}')
#brown = nltk.corpus.brown
#
#for sent in brown.tagged_sents():
#    tree = cp.parse(sent)
#    for subtree in tree.subtrees():
#        if subtree.label() == 'CHUNK': print(subtree)
#

class RegexExtractor:
    def __init__(self,
                 entity_name='Regex',
                 extraction_regex=[],
                 exclusion_offset=40,
                 exclusion_list=[],
                 inclusion_offset=40,
                 inclusion_list=[],
                 indicator=False,
                 integer_indicator=False,
                 unique_indicator=False,
                 length_limit=1000,
                 default_value=[],
                 remove_char="",
                 upper_case=False,
                 replace_char=["", ""]
                 ):

        self.entity_name = entity_name
        self.extraction_regex = extraction_regex
        self.search_pattern = re.compile(self.extraction_regex,
                                         re.VERBOSE | re.MULTILINE | re.IGNORECASE)

        self.exclusion_offset = exclusion_offset
        self.exclusion_list = exclusion_list
        self.inclusion_offset = inclusion_offset
        self.inclusion_list = inclusion_list
        self.indicator = indicator

        self.integer_indicator = integer_indicator
        self.unique_indicator = unique_indicator
        self.length_limit = length_limit
        self.remove_char = remove_char
        self.upper_case = upper_case
        self.replace_char = replace_char

        self.exclusion_pats = []
        for exc in self.exclusion_list:
            self.exclusion_pats.append(re.compile(exc, re.IGNORECASE))
        self.inclusion_pats = []
        for exc in self.inclusion_list:
            self.inclusion_pats.append(re.compile(exc, re.IGNORECASE))

        self.default_value = default_value

        '''        
        try:        
            if self.integer_indicator: 
                self.default_value = [int(eval(default_value))]
            else:
                self.default_value = [eval(default_value)] 
        except:
            self.default_value = []
        '''

    def get_matches(self, doc):
        '''
        Input: doc - string containing description text
        Returns: list of strings, each one is a valid phone number
        '''
        doc = doc["text"]
        res = []

        try:
            if self.integer_indicator:
                default_value = [int(eval(self.default_value))]
            else:
                default_value = [eval(self.default_value)]
        except:
            default_value = self.default_value

        for p in self.search_pattern.finditer(doc):
            found_exc = False
            found_inc = False
            start_pos, end_pos = p.span()

            # Seacrh through all exclusion list items and tag True if found in at least one of them
            for exc_pat in self.exclusion_pats:
                if exc_pat.search(doc[max(start_pos - self.exclusion_offset, 0):start_pos]):
                    found_exc = True

                    # Seacrh through all inclusion list items and tag True if found in at least one of them
            if not self.inclusion_list:
                found_inc = True
            else:
                for inc_pat in self.inclusion_pats:
                    if inc_pat.search(doc[max(start_pos - self.inclusion_offset, 0):start_pos]):
                        found_inc = True

            # If not found in any of the exclusion list items and found in the inclusion list items than append to extraction list
            if (not found_exc) and found_inc:
                if self.integer_indicator:
                    int_value = int(re.sub("[^0-9]", "", p.group()))
                    if len(str(int_value)) <= self.length_limit: res.append(int_value)
                else:
                    res.append(p.group().replace(self.remove_char, ""))
                    if self.upper_case:
                        res.append(re.sub(self.replace_char[0], self.replace_char[1],
                                          re.sub(self.remove_char, "", p.group().upper())))
                    else:
                        res.append(
                            re.sub(self.replace_char[0], self.replace_char[1], re.sub(self.remove_char, "", p.group())))

        # Filter to only unique entities in the extraction list
        res_uniq = list(set(res))

        if (len(res_uniq) < 1) or (self.unique_indicator and len(res_uniq) > 1):
            res_uniq = default_value

            # Return empty list if there's a demand for unique value and more than 1 value parsed
        # if (self.unique_indicator) and len(res_uniq)>1:
        #    dict = {self.entity_name:[]}
        #    return dict

        if self.indicator:
            dict = {self.entity_name: len(res_uniq) > 0}
            return dict

        dict = {self.entity_name: res_uniq}
        return dict