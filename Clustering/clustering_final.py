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