import re
import nltk
import pickle
import os, sys
import numpy as np
import collections as coll
from bisect import bisect_left

import unigrams as uni

# As corpus contains shakeshpeer, our life is truely hard
reload(sys)
sys.setdefaultencoding('utf-8')

SOS0 = uni.SOS0
EOS0 = uni.EOS0
SOS1 = uni.SOS1
EOS1 = uni.EOS1
unknown = uni.unknown
discount = 0.25

# To get index in vocab efficiently
def indexof(arr, elem):
    ii = bisect_left(arr, elem)
    if ii != len(arr) and arr[ii] == elem:
        return ii
    return -1

def generate_bigram_counts(text):
    bigram = {}

    # u given v -> p(u|v)
    for ii in range(len(text) - 1):
        uu, vv = text[ii + 1], text[ii]
        if vv == EOS0:
            continue
        if bigram.has_key(vv) == False:
            bigram[vv] = {}
        if bigram[vv].has_key(uu) == False:
            bigram[vv][uu] = 0
        bigram[vv][uu] += 1
    bigram[unknown] = {}
    return bigram


# Now that we have both bigram and unigrams, calculate probabilities

# Try 1: Discount smoothing
def katz_backoff2(bicnts, unicnts, vocab):
    # Class A: c(v, u) > 0
    # Class B: c(v, u) = 0
    cnt = 0
    voc_len = len(vocab)
    uni_total = sum(unicnts.values())
    for vv in bicnts.keys():
        pdist = bicnts[vv]
        balance = 0
        sumb = 0
        suma = 0

        # print cnt
        for uu in pdist.keys():
            balance += (pdist[uu] - discount) / unicnts[vv]
            suma += (unicnts[uu] - uni.discount)
        balance = 1 - balance
        sumb = uni_total - suma
        # cnt += 1
        bicnts[vv] = (pdist, balance, sumb)

def katz_backoff_prob2(biprobs, unicnts, vocab, vv, uu):
    base = vv
    tar = uu
    voc_len = len(vocab)
    if biprobs.has_key(vv) == False:
        base = unknown
    if indexof(vocab, uu) == -1:
        tar = unknown

    # print base, tar
    pdist = biprobs[base][0]
    balance = biprobs[base][1]
    sumb = biprobs[base][2]

    prob = 0.0
    if pdist.has_key(tar) == False:
        if tar == unknown:
            prob = balance * (uni.discount * (voc_len - 1)) / sumb
        else:
            prob = balance * (unicnts[tar] - uni.discount) / sumb
    else:
        prob = (pdist[tar] - discount) / unicnts[base]
    if prob >= 1:
        print 'Error'
    return prob

def run(fname = 'brown.train'):
    corpus_file = fname

    unifile = corpus_file.split('.')[0] + '_uniprobs'
    fp = open(unifile, 'r')
    unicnts = pickle.load(fp)
    fp.close()

    vocab = sorted(unicnts.keys())

    cfp = open(corpus_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()

    bicnts = generate_bigram_counts(text)
    # print 'Found', len(bicnts), 'bigrams.'
    katz_backoff2(bicnts, unicnts, vocab)

    bifile = corpus_file.split('.')[0] + '_biprobs'
    fp = open(bifile, 'w')
    pickle.dump(bicnts, fp)
    fp.close()
# run()
