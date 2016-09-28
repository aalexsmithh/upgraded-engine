'''
Created on Jul 14, 2015

@author: jcheung
'''

import sys, os, codecs
import sklearn
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn import svm, linear_model, naive_bayes 

# model parameters
n = 1
lemmatize = False
lowercase = True

stoplist = set(stopwords.words('english'))

################################################################################
# reading data set
def read_tac(year):
    '''
    Read data set and return feature matrix X and labels y.
    
    X - (ndocs x nfeats)
    Y - (ndocs)
    '''
	# modify this according to your directory structure
    sub_folder = '../data/tac%s' % year
    X, Y = [], []
    
    # labels
    labels_f = 'tac%s.labels' % year
    
    fh = open(os.path.join(sub_folder, labels_f))
    for line in fh:
        docid, label = line.split()
        Y.append(int(label))
    
    # tac 10
    if year == '2010':
        template = 'tac10-%04d.txt'
        s, e = 1, 921
    elif year == '2011':
        template = 'tac11-%04d.txt'
        s, e = 921, 1801
        
    for i in xrange(s, e):
        fname = os.path.join(sub_folder, template % i)
        X.append(extract_features(fname, n, lemmatize, lowercase))
    
    
    nfeats = 100 # TODO: you'll have to figure out how many features you need
    
    # convert indices to numpy array
    for j, x in enumerate(X):
        arr = np.zeros(nfeats)
        for index in X[j]:
            arr[index] += 1.0
        X[j] = arr

    Y = np.array(Y)
    X = np.vstack(tuple(X))
    return X, Y
        

################################################################################
# feature extraction

def ispunct(some_string):
    return not any(char.isalnum() for char in some_string)

def get_tokens(s):
    '''
    Tokenize into words in sentences.
    
    Returns list of strs
    '''
    retval = []
    sents = sent_tokenize(s)
    
    for sent in sents:
        tokens = word_tokenize(sent)
        retval.extend(tokens)
    return retval

def extract_features(f, n, lemmatize, lowercase):
    '''
    Extract features from text file f into a feature vector.
    
    n: maximum length of n-grams
    lemmatize: (boolean) whether or not to lemmatize
    lowercase: (boolean) whether or not to lowercase everything
    '''
    
    s = codecs.open(f, 'r', encoding = 'utf-8').read()
    s = codecs.encode(s, 'ascii', 'ignore')
    
    tokens = get_tokens(s)
    print tokens # This demonstrates that you are reading the tokens. You can comment it out or remove this line.
    indices = []
    # TODO: fill this part in
    return indices

################################################################################

# evaluation code
def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if int(gold[i]) == int(predict[i]):
            corr += 1
    acc = float(corr) / len(gold)
    print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)
################################################################################

if __name__ == '__main__':

	# main driver code
    X, Y = read_tac('2010')
    