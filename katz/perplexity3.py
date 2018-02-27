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

def run(corpus_file = 'gutenberg.train', test_file = 'gutenberg.test', dev_file = 'gutenberg.dev'):
    unifile = corpus_file.split('.')[0] + '_uniprobs'
    bifile = corpus_file.split('.')[0] + '_biprobs'
    trifile = corpus_file.split('.')[0] + '_triprobs'

    cfp = open(dev_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()

    cfp = open(test_file, 'r')
    ttext = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()

    fp = open(unifile, 'r')
    unicnts = pickle.load(fp)
    fp.close()

    fp = open(bifile, 'r')
    bicnts = pickle.load(fp)
    fp.close()

    fp = open(trifile, 'r')
    tricnts = pickle.load(fp)
    fp.close()

    vocab = sorted(unicnts.keys())

    optdisc = 0
    minval = np.inf
    bi.discount = 0.1
    print 'bigram'
    print 'Discount', 'Perplexity'
    for ii in range(9):
        perp1 = bigram_perplexity(text, bicnts, unicnts, vocab)
        if perp1 < minval:
            minval = perp1
            optdisc = bi.discount
        print bi.discount, perp1
        bi.discount += 0.1

    bi.discount = optdisc
    print 'Perplexity on test =', bigram_perplexity(ttext, bicnts, unicnts, vocab) , 'at', optdisc
    print '\n'

    print 'trigram'
    print 'discount', 'perplexity'
    optdisc = 0
    minval = np.inf
    tri.discount = 0.1
    for ii in range(9):
        perp1 = trigram_perplexity(text, tricnts, bicnts, unicnts, vocab)
        if perp1 < minval:
            minval = perp1
            optdisc = bi.discount
        print tri.discount, perp1
        tri.discount += 0.1
    tri.discount = optdisc
    perp1 = trigram_perplexity(ttext, tricnts, bicnts, unicnts, vocab)
    print 'Perplexity on test =', trigram_perplexity(ttext, tricnts, bicnts, unicnts, vocab) , 'at', optdisc

uni.discount = 0.5
uni.run('gutenberg.train')
uni.run('brown.train')
uni.run('unified.train')

bi.run('gutenberg.train')
bi.run('brown.train')
bi.run('unified.train')

tri.run('gutenberg.train')
tri.run('brown.train')
tri.run('unified.train')

print 'Done counting'
print 'S1'
run('gutenberg.train', 'gutenberg.test', 'gutenberg.dev')
print 'S2'
run('brown.train', 'brown.test', 'brown.dev')
print 'S3'
run('unified.train', 'gutenberg.test', 'gutenberg.dev')
print 'S4'
run('unified.train', 'brown.test', 'brown.dev')
