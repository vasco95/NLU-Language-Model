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
sys.setdefaultencoding('latin-1')

SOS0 = uni.SOS0
EOS0 = uni.EOS0
SOS1 = uni.SOS1
EOS1 = uni.EOS1
unknown = uni.unknown

def bigram_perplexity(text, biprobs, unicnts, vocab):
    perp = 0.0
    cnt = 0
    for ii in range(len(text) - 1):
        uu, vv = text[ii + 1], text[ii]
        if vv == EOS0 or vv == EOS1 or vv == SOS1:
            continue
        cnt += 1

        prob = bi.katz_backoff_prob2(biprobs, unicnts, vocab, vv, uu)
        perp += -1.0 * np.log(prob)
    return np.exp(perp / cnt)

def trigram_perplexity(text, tricnts, biprobs, unicnts, vocab):
    perp = 0.0
    cnt = 0
    for ii in range(len(text) - 2):
        ww, vv, uu = text[ii], text[ii + 1], text[ii + 2]
        if ww == EOS0 or ww == EOS1:
            continue
        cnt += 1

        prob = tri.katz_backoff_prob3(tricnts, biprobs, unicnts, vocab, ww, vv, uu)
        perp += -1.0 * np.log(prob)
    return np.exp(perp / cnt)

uni.discount = 0.1
bi.discount = 0.1
tri.discount = 0.1

fp = open('uniprobs', 'r')
unicnts = pickle.load(fp)
fp.close()
vocab = sorted(unicnts.keys())

corpus_file = 'brown.dev'
cfp = open(corpus_file, 'r')
text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
cfp.close()

optdisc = 0
minval = np.inf
print 'bigram'
print 'discount', 'perplexity'
for ii in range(9):
    bi.run()

    fp = open('biprobs', 'r')
    biprobs = pickle.load(fp)
    fp.close()

    perp1 = bigram_perplexity(text, biprobs, unicnts, vocab)
    if perp1 < minval:
        minval = perp1
        optdisc = bi.discount
    print bi.discount, perp1
    bi.discount += 0.1

print '\n'
bi.discount = optdisc
bi.run()
fp = open('biprobs', 'r')
biprobs = pickle.load(fp)
fp.close()

print 'trigram'
print 'discount', 'perplexity'
for ii in range(9):
    tri.run()

    fp = open('triprobs', 'r')
    tricnts = pickle.load(fp)
    fp.close()

    perp1 = trigram_perplexity(text, tricnts, biprobs, unicnts, vocab)
    print tri.discount, perp1
    tri.discount += 0.1
