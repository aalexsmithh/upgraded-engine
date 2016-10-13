from gensim.models import Doc2Vec, doc2vec
import numpy as np
from scipy import optimize
import csv, sys

class SVM(object):

	def __init__(self):
		pass

	def _loss(self):
		return 0.5 * np.square(np.linalg.norm(self.w))

	def _lagrangian(self):
		print self.a.shape, self.y.T.shape, (self.w * self.x.T).shape
		return self._loss() + (self.a * (1 - self.y.T * (self.w * self.x.T)))

	def fit(self,X,Y):
		self.x = np.asmatrix(X)
		self.y = np.asmatrix(Y)
		self.w = np.ones(self.x.shape[1])
		self.a = np.ones(self.x.shape[0])
		print self._lagrangian()

	def predict(self):
		pass

def main():
	DEF_BREAK_PT = 60000
	VECTOR_SIZE = 50
	labels_raw = openFromCSV('datasets/train_out.csv')
	labels = categorize(labels_raw)
	print 'Opening model...'
	model = Doc2Vec.load(str(DEF_BREAK_PT)+'_'+str(VECTOR_SIZE)+'.gen')
	print 'Fitting model...'
	X = getVectorsFromGensim(model,'train',0,1000)
	Y = labels[0:1000]


	cf = SVM()
	cf.fit(X,Y)

def categorize(raw_labels):
	labels = []
	for label in raw_labels:
		if label == 'math':
			labels.append(1)
		elif label == 'cs':
			labels.append(2)
		elif label == 'stat':
			labels.append(3)
		elif label == 'physics':
			labels.append(4)
		else:
			pass
	return labels

def getVectorsFromGensim(model,prefix,start,stop):
	vectors = []
	for i in range(start,stop):
		index = prefix + '_%s' % i
		vectors.append(model.docvecs[index])
	return vectors

def openFromCSV(file):
	open_input = []
	with open(file, 'rb') as csvfile:
		read = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in read:
			if row[1] == 'abstract' or row[1] == 'category':
				# print row[0], row[1], 'skipped'
				pass
			else:
				open_input.append(row[1])
	return open_input

if __name__ == '__main__':
	main()