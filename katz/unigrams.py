#!/usr/bin/python

# Unigram count calculations
import re
import nltk
import pickle
import os, sys
import collections as coll
from bisect import bisect_left

# As corpus contains shakeshpeer, our life is truely hard
reload(sys)
sys.setdefaultencoding('utf-8')

SOS0 = 'SOS0'
EOS0 = 'EOS0'
SOS1 = 'SOS1'
EOS1 = 'EOS1'
unknown = 'UNK'

corpus_file = 'brown.train'
discount = 0.25

def run(fname = 'brown.train'):
    corpus_file = fname
    cfp = open(corpus_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())

    # Get unigram counts. This also counts SOS, EOS which is not informative
    unicnts = coll.Counter(text)

    # Keeping dictionary for unigram probabilities.
    total = sum(unicnts.values())
    uniprobs = {}

    for entry in unicnts.most_common():
        word = entry[0]
        cnt = entry[1]
        uniprobs[word] = cnt
        total += cnt

    if unicnts.has_key(unknown) == True:
        print 'Unknown present in corpus.'
        sys.exit(1)

    uniprobs[unknown] = 0

    # Dump these counts to a file
    unifile = corpus_file.split('.')[0] + '_uniprobs'
    fp = open(unifile, 'w')
    pickle.dump(uniprobs, fp)
    fp.close()
# run()
