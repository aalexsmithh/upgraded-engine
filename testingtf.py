import tensorflow as tf
import csv, sys
from gensim.models import Doc2Vec, doc2vec

def main():
	csv.field_size_limit(sys.maxsize)
	DEF_BREAK_PT = 88000
	VECTOR_SIZE = 400

	labels_raw = openFromCSV('datasets/train_out.csv')
	labels = categorize(labels_raw)

	sess = tf.InteractiveSession()
	x = tf.placeholder(tf.float32, shape=[None, VECTOR_SIZE])
	y_ = tf.placeholder(tf.float32, shape=[None, 4])

	W = tf.Variable(tf.zeros([VECTOR_SIZE,4]))
	b = tf.Variable(tf.zeros([4]))

	sess.run(tf.initialize_all_variables())

	y = tf.matmul(x,W) + b

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	model = Doc2Vec.load(str(DEF_BREAK_PT)+'_'+str(VECTOR_SIZE)+'.gen')

	for i in range(1000):
		index = DEF_BREAK_PT/1000
		X_train = getVectorsFromGensim(model,'train',(index*i),(index*i+index))
		Y_train = labels[(index*i):(index*i+index)]
		train_step.run(feed_dict={x: X_train, y_: Y_train})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	Y_test = labels[DEF_BREAK_PT:]
	X_test = getVectorsFromGensim(model,'test',DEF_BREAK_PT,DEF_BREAK_PT+len(Y_test))

	print(accuracy.eval(feed_dict={x: X_test, y_: Y_test}))

def categorize(raw_labels):
	labels = []
	for label in raw_labels:
		if label == 'math':
			labels.append([1,0,0,0])
		elif label == 'cs':
			labels.append([0,1,0,0])
		elif label == 'stat':
			labels.append([0,0,1,0])
		elif label == 'physics':
			labels.append([0,0,0,1])
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