import re
import nltk
import pickle
import os, sys
import collections as coll
from bisect import bisect_left

import unigrams as uni
import bigrams as bi
import trigrams as tri

uni.run('gutenberg.train')
uni.run('brown.train')
uni.run('unified.train')

bi.run('gutenberg.train')
bi.run('brown.train')
bi.run('unified.train')

tri.run('gutenberg.train')
tri.run('brown.train')
tri.run('unified.train')
