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
discount2 = 0.4
discount3 = 0.25

total2 = 0
unique_tri = 0

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

        # if vv == EOS1 or vv == EOS0 or vv == SOS1:
        if vv == EOS1:
            continue

        if bigram.has_key(vv) == False:
            bigram[vv] = {}
        if bigram[vv].has_key(uu) == False:
            bigram[vv][uu] = 0
            if vv != EOS0 and vv != SOS1:
                total += 1
        bigram[vv][uu] += 1
    bigram[unknown] = {}
    unigram[unknown] = 0
    return unigram, bigram, total

def generate_trigram_counts(text):
    trigram = {}
    global total2
    total2 = 0

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
            total2 += 1
        trigram[ww][vv][uu] += 1

    for ww in trigram.keys():
        trigram[ww][unknown] = {}
    trigram[unknown] = {}
    trigram[unknown][unknown] = {}
    # print len(trigram)
    return trigram, total

def knn_smooth2(bicnts):
    kbicnts = {}
    for vv in bicnts.keys():
        for uu in bicnts[vv].keys():
            if kbicnts.has_key(uu) == False:
                kbicnts[uu] = 0.0
            kbicnts[uu] += 1
    return kbicnts

def knn_smooth3(tricnts):
    ktricnts = {}
    for ww in tricnts.keys():
        for vv in tricnts[ww].keys():
            if ktricnts.has_key(vv) == False:
                ktricnts[vv] = {}
            for uu in tricnts[ww][vv].keys():
                if ktricnts[vv].has_key(uu) == False:
                    ktricnts[vv][uu] = 0.0
                ktricnts[vv][uu] += 1
    return ktricnts

def knn_prob2(bicnts, unicnts, total, kbicnts, ktricnts, vv, uu, vocab):
    base = vv
    tar = uu
    global total2

    idx1 = indexof(vocab, vv)
    idx2 = indexof(vocab, uu)
    if idx1 == -1 and idx2 == -1:
        prob = (discount1 * (len(vocab) - 1)) / total
        return prob

    if idx1 == -1 and idx2 != -1:
        ncnt = kbicnts[uu]
        prob = (ncnt * 1.0 - discount1) / total
        return prob

    if indexof(vocab, uu) == -1:
        tar = unknown

    prob = 0.0
    if ktricnts[base].has_key(tar) == False:
    else:
        ncnt = ktricnts[base][tar] - discount2
        biprob = ncnt / total2
        balance = total * discount2 / total2

    return prob

def knn_prob3(tricnts, bicnts, unicnts, total, kbicnts, ktricnts, ww, vv, uu, vocab):
    base0 = ww
    base1 = vv
    tar = uu

    idx0 = indexof(vocab, ww)
    idx1 = indexof(vocab, vv)
    idx2 = indexof(vocab, uu)

    if idx0 == -1:
        return knn_prob2(bicnts, unicnts, total, kbicnts, ktricnts, base1, tar, vocab)
    else:
        ncnt = 0
        triprob = 0.0
        prob = 0

        if tricnts[base0].has_key(base1) == False:
            ncnt = 0
        elif tricnts[base0][base1].has_key(tar) == False:
            ncnt = 0
        else:
            ncnt = tricnts[base0][base1][tar]

        ncnt = ncnt - discount3
        if ncnt < 0.0:
            biprob = knn_prob2(bicnts, unicnts, total, kbicnts, ktricnts, base1, tar, vocab)
            if tricnts[base0].has_key(base1) == False:
                prob = biprob
            else:
                balance = discount3 * len(tricnts[base0][base1]) / bicnts[base0][base1]
                prob = balance * biprob
        else:
            # print base0, base1
            triprob = ncnt / bicnts[base0][base1]
            balance = discount3 * len(tricnts[base0][base1]) / bicnts[base0][base1]
            prob = triprob + balance * knn_prob2(bicnts, unicnts, total, kbicnts, ktricnts, base1, tar, vocab)
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

def trigram_perplexity(text, tricnts, bicnts, unicnts, total, kbicnts, ktricnts, vocab):
    perp = 0.0
    cnt = 0
    for ii in range(len(text) - 2):
        ww, vv, uu = text[ii], text[ii + 1], text[ii + 2]
        if ww == EOS0 or ww == EOS1:
            continue
        cnt += 1

        prob = knn_prob3(tricnts, bicnts, unicnts, total, kbicnts, ktricnts, ww, vv, uu, vocab)
        perp += -1.0 * np.log(prob)
    return np.exp(perp / cnt)

def trigram_token_generate(tricnts, bicnts, unicnts, total, kbicnts, ktricnts, vocab):
    ww, vv = SOS1, SOS0
    ret = ''

    while vv != EOS0:
        # Generate probabilities and values
        probs = []
        vals = []
        for uu in vocab:
            if uu != SOS0 and uu != SOS1 and uu != EOS1:
                vals.append(uu)
                calc = knn_prob3(tricnts, bicnts, unicnts, total, kbicnts, ktricnts, ww, vv, uu, vocab)
                probs.append(calc)

        uu = np.random.choice(vals, 1, p = probs)
        while uu[0] == unknown or uu[0] == EOS1 or uu[0] == SOS1 or uu[0] == SOS0:
            uu = np.random.choice(vals, 1, p = probs)
        uu = uu[0]
        if uu != EOS0:
            ret = ret + ' ' + uu
        ww, vv = vv, uu
    print ret, '\n'

def run(corpus_file = 'gutenberg.train', test_file = 'gutenberg.test', dev_file = 'gutenberg.dev'):
    cfp = open(corpus_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()

    unicnts, bicnts, total = generate_counts(text)
    vocab = sorted(unicnts.keys())
    kbicnts = knn_smooth2(bicnts)
    tricnts = generate_trigram_counts(text)
    ktricnts = knn_smooth3(tricnts)

    cfp = open(dev_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()
    global discount3
    discount3 = 0.1
    minval = np.inf
    opt = 0.1
    print 'discount', 'perplexity'
    for ii in range(1, 10):
        discount3 = 0.1 * ii
        perp1 = trigram_perplexity(text, tricnts, bicnts, unicnts, total, kbicnts, ktricnts, vocab)
        if perp1 < minval:
            minval = perp1
            opt = discount2
        print discount3, perp1

    discount3 = opt
    cfp = open(test_file, 'r')
    text = re.findall(r"[\w']+|[.,!?;]", cfp.read())
    cfp.close()
    perp1 = trigram_perplexity(text, tricnts, bicnts, unicnts, total, kbicnts, ktricnts, vocab)
    print 'Perplexity on test =', perp1, 'at', opt

    trigram_token_generate(tricnts, bicnts, unicnts, total, kbicnts, ktricnts, vocab)

print 'S1'
run('gutenberg.train', 'gutenberg.test', 'gutenberg.dev')
# print 'S2'
# run('brown.train', 'brown.test', 'brown.dev')
# print 'S3'
# run('unified.train', 'gutenberg.test', 'gutenberg.dev')
# print 'S4'
# run('unified.train', 'brown.test', 'brown.dev')
