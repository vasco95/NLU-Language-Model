#!/usr/bin/python

# This file combines all the documents in a given corpus to give a single mega
# file. Additionally a for each sentence SOS and EOS are added.
#
import re
import nltk
import os, sys
import numpy as np
from nltk.corpus import brown
from nltk.corpus import gutenberg

# As corpus contains shakeshpeer, our life is truely hard
reload(sys)
sys.setdefaultencoding('utf-8')

SOS0 = 'SOS0'
EOS0 = 'EOS0'
SOS1 = 'SOS1'
EOS1 = 'EOS1'

prefix = SOS1 + ' ' + SOS0
postfix = EOS0 + ' ' + EOS1
# postfix = EOS0

def create_corpus(dir_name = 'brown'):
    corpus = brown
    if dir_name == 'gutenberg':
        corpus = gutenberg

    sentences = []
    for entry in corpus.sents():
        stmp = ''
        for word in entry:
            stmp += word + ' '
        sentences.append(stmp);

    idx = np.random.randint(len(sentences), size = len(sentences))
    train_len = int(len(sentences) * 0.8)
    devset_len = int(len(sentences) * 0.1)

    fout = open(dir_name + '.train', 'w')
    for ii in range(train_len):
        entry = sentences[idx[ii]]
        entry = entry.replace('\n', ' ')
        entry = ' ' + prefix + ' ' + entry + ' ' + postfix + ' '
        fout.write(entry)

    fout = open(dir_name + '.dev', 'w')
    for ii in range(train_len, (train_len + devset_len)):
        entry = sentences[idx[ii]]
        entry = entry.replace('\n', ' ')
        entry = ' ' + prefix + ' ' + entry + ' ' + postfix + ' '
        fout.write(entry)
    fout.close()

    fout = open(dir_name + '.test', 'w')
    for ii in range(devset_len + train_len, len(sentences)):
        entry = sentences[idx[ii]]
        entry = entry.replace('\n', ' ')
        entry = ' ' + prefix + ' ' + entry + ' ' + postfix + ' '
        fout.write(entry)
    fout.close()

create_corpus('gutenberg')
print 'Created gutenberg'
create_corpus('brown')
print 'Created brown'
cp1 = open('gutenberg.train', 'r')
cp2 = open('brown.train', 'r')
text = cp1.read() + ' ' + cp2.read()
cp1.close()
cp2.close()
fp = open('unified.train', 'w')
fp.write(text);
fp.close()
