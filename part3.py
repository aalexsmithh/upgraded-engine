# from gensim.models import tfidfmodel
from gensim.models import Doc2Vec, doc2vec
from gensim import utils, matutils
from sklearn import linear_model, svm
from random import shuffle
import csv, sys, multiprocessing, cPickle, time

def main():
	DEF_BREAK_PT = 60000
	VECTOR_SIZE = 50
	csv.field_size_limit(sys.maxsize)
	cores = multiprocessing.cpu_count()

	sources = {'datasets/train_in.csv':'train','datasets/test_in.csv':'test'}

	print 'Organizing input data ...'
	documents = LabeledDocuments(sources)
	labels_raw = openFromCSV('datasets/train_out.csv')
	labels = categorize(labels_raw)

	# print 'Training gensim model on %s cores...' % cores
	# model = Doc2Vec(workers=cores,size=VECTOR_SIZE,min_count=1)
	# print '\t Building vocab...'
	# model.build_vocab(documents.to_array())
	# for epoch in range(20):
	# 	start = time.time()
	# 	print '\t epoch', epoch,
	# 	model.train(documents.documents_perm())
	# 	print '- %s' % (time.time() - start)
	
	# print 'Saving model...'
	# model.save(open(str(DEF_BREAK_PT)+'_'+str(VECTOR_SIZE)+'.gen','wb'))

	############## IF MODEL IS ALREADY TRAINED
	print 'Opening model...'
	model = Doc2Vec.load(str(DEF_BREAK_PT)+'_'+str(VECTOR_SIZE)+'.gen')

	print 'Fitting model...'
	X_train = getVectorsFromGensim(model,'train',0,DEF_BREAK_PT)
	Y_train = labels[0:DEF_BREAK_PT]

	# svm.SVC()
	# linear_model.LogisticRegression(solver='lbfgs')
	classifier = trainClassificationModel(X_train,Y_train,linear_model.LogisticRegression(solver='lbfgs'))

	print 'Saving classifier...'
	with open('50_lr.pkl', 'wb') as fid:
		cPickle.dump(classifier, fid)

	print 'Testing model...'
	Y_test = labels[DEF_BREAK_PT:]
	X_test = getVectorsFromGensim(model,'train',DEF_BREAK_PT,DEF_BREAK_PT+len(Y_test))
	print classifier.score(X_test,Y_test)

def preprocessData(data,label=None):
	out = []
	if label is not None:
		for i in range(0,len(data)):
			if data[i] == 'abstract': #and label[i] == 0:
				# print i, data[i], label[i]
				pass
			else:
				test = doc2vec.TaggedDocument(utils.simple_preprocess(data[i]),label[i])
				out.append(test)
	else:
		for i in range(0,len(data)):
			if data[i] == 'abstract': #and label[i] == 0:
				# print i, data[i], label[i]
				pass
			else:
				test = utils.simple_preprocess(data[i])
				out.append(test)
	return out

def trainClassificationModel(X,Y,model):
	model.fit(X,Y)
	return model

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

class LabeledDocuments(object):
	def __init__(self, sources):
		self.sources = sources
		
		flipped = {}
		
		# make sure that keys are unique
		for key, value in sources.items():
			if value not in flipped:
				flipped[value] = [key]
			else:
				raise Exception('Non-unique prefix encountered')
	
	def __iter__(self):
		for source, prefix in self.sources.items():
			raw_data = openFromCSV(source)
			for item_no, line in enumerate(raw_data):
				data = utils.simple_preprocess(line,deacc=True,max_len=30)
				yield doc2vec.TaggedDocument(data, [prefix + '_%s' % item_no])
	
	def to_array(self):
		self.documents = []
		for source, prefix in self.sources.items():
			raw_data = openFromCSV(source)
			for item_no, line in enumerate(raw_data):
				data = utils.simple_preprocess(line,deacc=True,max_len=30)
				self.documents.append(doc2vec.TaggedDocument(data, [prefix + '_%s' % item_no]))
		return self.documents
	
	def documents_perm(self):
		shuffle(self.documents)
		return self.documents

if __name__ == '__main__':
	main()