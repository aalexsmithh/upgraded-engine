import tensorflow as tf

def main():
	VECTOR_SIZE = 400
	CATEGORIES = 4

	sess = tf.InteractiveSession()
	x = tf.placeholder(tf.float32, shape=[None, VECTOR_SIZE])
	y_ = tf.placeholder(tf.float32, shape=[None, CATEGORIES])

	sess.run(tf.initialize_all_variables())

	y = tf.matmul(x,W) + b

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	for i in range(1000):
		train_step.run(feed_dict={x: X_train, y_: Y_train})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print(accuracy.eval(feed_dict={x: X_test, y_: Y_test}))


if __name__ == '__main__':
	main()