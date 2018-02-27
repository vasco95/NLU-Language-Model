import re
import nltk
import pickle
import os, sys
import numpy as np
import collections as coll
from bisect import bisect_left

import unigrams as uni
import bigrams as bi
import trigrams as tri

# As corpus contains shakeshpeer, our life is truely hard
reload(sys)
sys.setdefaultencoding('utf-8')

SOS0 = uni.SOS0
EOS0 = uni.EOS0
SOS1 = uni.SOS1
EOS1 = uni.EOS1
unknown = uni.unknown

def bigram_token_generate(bicnts, unicnts, vocab):
    vv = SOS0
    ret = ''

    while vv != EOS0:
        # Generate probabilities and values
        bicnt = bicnts[vv]

        pdist = bicnt[0]
        balance = bicnt[1]
        sumb = bicnt[2]
        probs = []
        vals = []

        # sum2 = 0
        # for uu in pdist.keys():
        #     sum2 += (unicnts[uu] - uni.discount)
        # sum1 = 0
        # for uu in vocab:
        #     if uu not in pdist.keys():
        #         vals.append(uu)
        #         calc = bi.katz_backoff_prob2(bicnts, unicnts, vocab, vv, uu)
        #         probs.append(calc)
        #         if uu == unknown:
        #             sum1 += (len(vocab) - 1) * uni.discount
        #         else:
        #             sum1 += (unicnts[uu] - uni.discount)
        #
        # print sum1, sum(probs), sumb, sum2, (sum(unicnts.values()) - sum2)

        for uu in vocab:
            vals.append(uu)
            calc = bi.katz_backoff_prob2(bicnts, unicnts, vocab, vv, uu)
            probs.append(calc)

#        print sum(probs)
        vv = np.random.choice(vals, 1, p = probs)
        while vv[0] == unknown or vv[0] == EOS1 or vv[0] == SOS1 or vv[0] == SOS0:
            vv = np.random.choice(vals, 1, p = probs)
        vv = vv[0]
        if vv != EOS0:
            ret = ret + ' ' + vv
    print ret, '\n'

def trigram_token_generate(tricnts, bicnts, unicnts, vocab):
    ww, vv = SOS1, SOS0
    ret = ''
    tok_cnt = 0

    while tok_cnt < 10:
        # Generate probabilities and values
        if tricnts.has_key(ww) == False:
            ww = unknown
        if tricnts[ww].has_key(vv) == False:
            vv = unknown

        tricnt = tricnts[ww][vv]

        pdist = tricnt[0]
        balance = tricnt[1]
        sumb = tricnt[2]
        probs = []
        vals = []

        for uu in vocab:
            vals.append(uu)
            calc = tri.katz_backoff_prob3(tricnts, bicnts, unicnts, vocab, ww, vv, uu)
            probs.append(calc)

#        print sum(probs)
        uu = np.random.choice(vals, 1, p = probs)
        while uu[0] == unknown or uu[0] == EOS1 or uu[0] == SOS1 or uu[0] == SOS0 or uu[0].isdigit() == True:
            uu = np.random.choice(vals, 1, p = probs)
        uu = uu[0]
        if uu != EOS0:
            tok_cnt += 1
            ret = ret + ' ' + uu
            ww, vv = vv, uu
        else:
            ww, vv = SOS1, SOS0

    print ret, '\n'

corpus_file = 'gutenberg.train'
unifile = corpus_file.split('.')[0] + '_uniprobs'
bifile = corpus_file.split('.')[0] + '_biprobs'
trifile = corpus_file.split('.')[0] + '_triprobs'

fp = open(unifile, 'r')
unicnts = pickle.load(fp)
fp.close()

vocab = sorted(unicnts.keys())

fp = open(bifile, 'r')
bicnts = pickle.load(fp)
fp.close()

fp = open(trifile, 'r')
tricnts = pickle.load(fp)
fp.close()

#bigram_token_generate(bicnts, unicnts, vocab)
trigram_token_generate(tricnts, bicnts, unicnts, vocab)
