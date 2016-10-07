import csv, sys, nltk, pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import bigrams, trigrams
import numpy as np
from scipy import sparse, io
import test

def create_features(do_traindata,do_labels):
    train_data = []
    train_label = []
    csv.field_size_limit(sys.maxsize)
    first = False
    if do_traindata:
        with open('datasets/train_in.csv', 'rb') as csvfile:
            read = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in read:
                if not first:
                    train_data.append(row[1])
                first = False
        feature = extract_features()
        first = True
    if do_labels:
        with open('datasets/train_out.csv', 'rb') as csvfile:
            read = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in read:
                if not first:
                    train_label.append(row[1])
                first = False
        labels = categorize(train_label)
        first = True

    print 'dumping'
    pickle.dump(feature,open('','wb'))
    print len(feature)

def get_feature_for_csv(input_data_csv,input_labels_csv,n,lower,lemma):
    csv.field_size_limit(sys.maxsize)
    X, Y = [], []

    print 'load features', '...',
    all_features = pickle.load(open('features_uni_lemma_lower.pickle','rb'))
    
    print 'extract for each document', '...',
    with open(input_data_csv, 'rb') as csvfile:
        read_X = [row for row in csv.reader(csvfile, delimiter=',', quotechar='|')]
        for row in read_X[0:1000]:
            X.append(extract_features(row[1],n,lower,lemma,all_features))
    csvfile.close()
    print 'assemble feature array', '...'
    for j, x in enumerate(X):
        arr = np.zeros(len(all_features))
        for index in X[j]:
            arr[index] += 1.0
        X[j] = arr
    X = np.vstack(tuple(X))

    with open(input_labels_csv, 'rb') as csvfile:
        read_Y = [row for row in csv.reader(csvfile, delimiter=',', quotechar='|')]
        train_label = []
        for row in read_Y[0:1000]:
            train_label.append(row[1])
        Y = categorize(train_label)
    del train_label, 
    return X, np.asarray(Y)

def extract_features(input_docs,n,lower,lemma,input_features):
    features = []
    indices = []
    if lower:
        input_docs = input_docs.lower()
    tokens = nltk.word_tokenize(input_docs)
    if lemma:
        w = nltk.WordNetLemmatizer()
        tokens = [w.lemmatize(token) for token in tokens]
    if n == 1:
        features = np.unique(np.asarray(tokens,dtype=str)).tolist()
    if n == 2:
        features = bigrams(tokens)

    for doc_feat in features:
        try:
            indices.append(input_features.index(doc_feat))
            # print indices[len(indices)-1:]
        except Exception, e:
            # print e
            pass
    return indices

def categorize(raw_labels):
    labels = [0] * len(raw_labels)
    index = 0
    for label in raw_labels:
        if label == 'math':
            labels[index] = 1
        elif label == 'cs':
            labels[index] = 2
        elif label == 'stat':
            labels[index] = 3
        elif label == 'physics':
            labels[index] = 4
        else:
            print label, index
            labels[index] = 0
        index += 1
    return labels

def create_feature_template(input_corpus,n,lower,lemma):
    input_data = []
    first = True
    with open(input_corpus, 'rb') as csvfile:
            read = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in read:
                if not first:
                    input_data = input_data + row[1]
                first = False
    if lower:
        print 'lowering'
        print 'tokenizing'
        tokens = nltk.word_tokenize(input_data.lower())
    else:
        print 'tokenizing'
        tokens = nltk.word_tokenize(input_data)
    if lemma:
        print 'lemmatizing'
        w = nltk.WordNetLemmatizer()
        tokens = [w.lemmatize(token) for token in tokens]
    feature = []
    if n == 1:
        print 'unigram'
        feature = np.unique(np.asarray(tokens,dtype=str)).tolist()
        # for token in tokens:
        #     if token not in feature:
        #         feature.append(token)
    if n == 2:
        print 'bigram'
        feature = bigrams(tokens)
    if n == 3:
        print 'trigram'
        feature = trigrams(tokens)
    return sorted(set(feature))

def main():
    X, Y = get_feature_for_csv('datasets/train_in.csv','datasets/train_out.csv',1,lower=True,lemma=True)
    print X.shape, Y.shape
    # np.savez_compressed('X_a.npz', sparse.csr_matrix(X))
    # np.savez_compressed('Y_a.npz', sparse.csr_matrix(Y))
    # np.savetxt('X_a.gz',X)
    # np.savetxt('Y_a.gz',Y)
    # io.mmwrite("X_b.mtx",sparse.csr_matrix(X))
    # io.mmwrite("Y_b.mtx",sparse.csr_matrix(Y))
    test.saveArray(X,Y,'lol')

if __name__ == '__main__':
    # create_feature_template('datasets/train_in.csv',1,True,True)
    main()