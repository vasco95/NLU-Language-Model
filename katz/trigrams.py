import re
import nltk
import pickle
import os, sys
import numpy as np
import collections as coll
from bisect import bisect_left

import unigrams as uni
import bigrams as bi

# As corpus contains shakeshpeer, our life is truely hard
reload(sys)
sys.setdefaultencoding('utf-8')

SOS0 = uni.SOS0
EOS0 = uni.EOS0
SOS1 = uni.SOS1
EOS1 = uni.EOS1
unknown = uni.unknown
discount = 0.25

def generate_trigram_counts(text):
    trigram = {}

    # u given w, v -> p(u|w v)
    for ii in range(len(text) - 2):
        uu, vv, ww = text[ii + 2], text[ii + 1], text[ii]
        if ww == EOS0 or ww == EOS1:
            continue
        if trigram.has_key(ww) == False:
            trigram[ww] = {}
        if trigram[ww].has_key(vv) == False:
            trigram[ww][vv] = {}
        if trigram[ww][vv].has_key(uu) == False:
            trigram[ww][vv][uu] = 0
        trigram[ww][vv][uu] += 1

    for ww in trigram.keys():
        trigram[ww][unknown] = {}
    trigram[unknown] = {}
    trigram[unknown][unknown] = {}
    # print len(trigram)
    return trigram

# Now that we have both trigram, bigram and unigrams, calculate probabilities

# Try 1: Discount smoothing
def katz_backoff3(tricnts, bicnts, unicnts, vocab):
    # Class A: c(w, v, u) > 0
    # Class B: c(w, v, u) = 0
    cnt = 0
    voc_len = len(vocab)
    # print len(tricnts)
    for ww in tricnts.keys():
        bicnt = bicnts[ww]
        # print cnt
        cnt += 1
        for vv in tricnts[ww].keys():
            pdist = tricnts[ww][vv]
            balance = 0
            sumb = 0
            suma = 0

            for uu in pdist.keys():
                balance += (pdist[uu] - discount) / bicnt[0][vv]
                suma += bi.katz_backoff_prob2(bicnts, unicnts, vocab, vv, uu)
            balance = 1 - balance
            sumb = 1 - suma
            tricnts[ww][vv] = (pdist, balance, sumb)

def katz_backoff_prob3(tricnts, bicnts, unicnts, vocab, ww, vv, uu):
    base0 = ww
    base1 = vv
    tar = uu
    voc_len = len(vocab)
    flag = False

    if tricnts.has_key(ww) == False:
        base0 = unknown
    if tricnts[base0].has_key(vv) == False:
        base1 = unknown

    # if tricnts[base0][base1][0].has_key(uu) == False:
    #     tar = unknown
    # print base0, base1
    # Backoff to bigrmas
    # if flag == True:
    #     balance = tricnts[unknown][unknown][1]
    #     sumb = tricnts[unknown][unknown][2]
    #     prob = balance * (bi.katz_backoff_prob2(bicnts, unicnts, vocab, vv, ww) / sumb)
    #     return prob
    #
    # if tricnts[base0].has_key(base1) == False:
    #     base1 = unknown
    #     base0 = unknown

    bicnt = bicnts[base0]
    pdist = tricnts[base0][base1][0]
    balance = tricnts[base0][base1][1]
    sumb = tricnts[base0][base1][2]

    prob = 0.0
    if pdist.has_key(tar) == False:
        prob = balance * (bi.katz_backoff_prob2(bicnts, unicnts, vocab, base1, tar) / sumb)
    else:
        prob = (pdist[tar] - discount) / bicnt[0][base1]
    return prob

def run(fname = 'brown.train'):
    corpus_file = fname

    unifile = corpus_file.split('.')[0] + '_uniprobs'
    fp = open(unifile, 'r')
    unicnts = pickle.load(fp)
    fp.close()

    vocab = sorted(unicnts.keys())

    bifile = corpus_file.split('.')[0] + '_biprobs'
    fp = open(bifile, 'r')
    bicnts = pickle.load(fp)
    fp.close()

    cfp = open(corpus_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()

    tricnts = generate_trigram_counts(text)
    katz_backoff3(tricnts, bicnts, unicnts, vocab)

    # print biprobs['jawbone'], unicnts['of']
    trifile = corpus_file.split('.')[0] + '_triprobs'
    fp = open(trifile, 'w')
    pickle.dump(tricnts, fp)
    fp.close()
# run()
