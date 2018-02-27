import re
import nltk
import pickle
import os, sys
import numpy as np
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

discount1 = 0.5
discount2 = 0.75

# To get index in vocab efficiently
def indexof(arr, elem):
    ii = bisect_left(arr, elem)
    if ii != len(arr) and arr[ii] == elem:
        return ii
    return -1

def generate_counts(text):
    bigram = {}
    unigram = {}
    total = 0

    # u given v -> p(u|v)
    for ii in range(len(text) - 1):
        uu, vv = text[ii + 1], text[ii]

        if unigram.has_key(vv) == False:
            unigram[vv] = 0
        unigram[vv] += 1

        if vv == EOS1 or vv == SOS1 or vv == EOS0:
            continue

        if bigram.has_key(vv) == False:
            bigram[vv] = {}
        if bigram[vv].has_key(uu) == False:
            bigram[vv][uu] = 0
            total += 1
        bigram[vv][uu] += 1
    bigram[unknown] = {}
    unigram[unknown] = 0
    return unigram, bigram, total

def knn_smooth2(bicnts):
    kbicnts = {}
    for vv in bicnts.keys():
        for uu in bicnts[vv].keys():
            if kbicnts.has_key(uu) == False:
                kbicnts[uu] = 0.0
            kbicnts[uu] += 1
    return kbicnts

def knn_prob2(bicnts, unicnts, total, kbicnts, vv, uu, vocab):
    base = vv
    tar = uu

    idx1 = indexof(vocab, vv)
    idx2 = indexof(vocab, uu)
    if idx1 == -1 and idx2 == -1:
        prob = (discount1 * (len(vocab) - 4)) / total
        return prob

    if idx1 == -1 and idx2 != -1:
        ncnt = kbicnts[uu]
        prob = (ncnt * 1.0 - discount1) / total
        return prob

    if indexof(vocab, uu) == -1:
        tar = unknown

    balance = (discount2 * len(bicnts[base])) / unicnts[base]
    prob = 0.0
    if tar != unknown:
        ncnt = kbicnts[tar]
        uniprob = (ncnt * 1.0 - discount1) / total

        if bicnts[base].has_key(tar) == False:
            ncnt = 0
        else:
            ncnt = bicnts[base][tar]
        ncnt = ncnt - discount2
        biprob = 0.0
        if ncnt < 0:
            biprob = 0.0
        else:
            biprob = (ncnt * 1.0) / unicnts[base]
        prob = biprob + uniprob * balance
    else:
        uniprob = (discount1 * (len(vocab) - 4)) / total
        prob = balance * uniprob

    return prob

def bigram_perplexity(text, biprobs, unicnts, total, kbicnts, vocab):
    perp = 0.0
    cnt = 0
    for ii in range(len(text) - 1):
        uu, vv = text[ii + 1], text[ii]
        if vv == EOS0 or vv == EOS1 or vv == SOS1:
            continue
        cnt += 1

        prob = knn_prob2(biprobs, unicnts, total, kbicnts, vv, uu, vocab)
        perp += -1.0 * np.log(prob)
    return np.exp(perp / cnt)

def run(corpus_file = 'gutenberg.train', test_file = 'gutenberg.test', dev_file = 'gutenberg.dev'):
    cfp = open(corpus_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()

    # perp1 = bigram_perplexity(text, bicnts, unicnts, total, kbicnts, vocab)
    # print 'Bigram on train =', perp1

    unicnts, bicnts, total = generate_counts(text)
    vocab = sorted(unicnts.keys())
    kbicnts = knn_smooth2(bicnts)

    cfp = open(dev_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()
    global discount2
    discount2 = 0.1
    minval = np.inf
    opt = 0.1
    print 'discount', 'perplexity'
    for ii in range(1, 10):
        discount2 = 0.1 * ii
        perp1 = bigram_perplexity(text, bicnts, unicnts, total, kbicnts, vocab)
        if perp1 < minval:
            minval = perp1
            opt = discount2
        print discount2, perp1

    discount2 = opt
    cfp = open(test_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()
    perp1 = bigram_perplexity(text, bicnts, unicnts, total, kbicnts, vocab)
    print 'Perplexity on test =', perp1, 'at', opt

print 'S1'
run('gutenberg.train', 'gutenberg.test', 'gutenberg.dev')
print 'S2'
run('brown.train', 'brown.test', 'brown.dev')
print 'S3'
run('unified.train', 'gutenberg.test', 'gutenberg.dev')
print 'S4'
run('unified.train', 'brown.test', 'brown.dev')
