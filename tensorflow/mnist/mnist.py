import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_input = 784
n_hidden1 = 100
n_hidden2 = 100
n_output = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([n_input, n_hidden1], -1.0, 1.0))
B1 = tf.Variable(tf.zeros([n_hidden1]))

W2 = tf.Variable(tf.random_uniform([n_hidden1, n_hidden2], -1.0, 1.0))
B2 = tf.Variable(tf.zeros([n_hidden2]))

W3 = tf.Variable(tf.random_uniform([n_hidden2, n_output], -1.0, 1.0))
B3 = tf.Variable(tf.zeros([n_output]))

H1 = tf.nn.relu(tf.matmul(X, W1) + B1)
H1 = tf.nn.dropout(H1, 0.9)
H2 = tf.nn.relu(tf.matmul(H1, W2) + B2)
H2 = tf.nn.dropout(H2, 0.9)
Ows = tf.matmul(H2, W3) + B3
O = tf.nn.softmax(tf.matmul(H2, W3) + B3)
# O = predict()
# pred = predict(false)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Ows, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(O), tf.float32), y), tf.float32))
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(O, 1), tf.argmax(y, 1)), tf.float32))

# costs = []

init = tf.global_variables_initializer()

print ("Start to learn")

with tf.Session() as sess:
	sess.run(init)

	epochs = 20
	batch_size = 128
	t_train = time.clock_gettime(0)
	for epoch in range(epochs):
		t_epoch = time.clock_gettime(0)
		for _ in range(int(len(mnist.train.images) / batch_size)):
			xs, ys = mnist.train.next_batch(batch_size) 
			# sess.run(optimizer, feed_dict = {X: mnist.train.images, y: mnist.train.labels})
			sess.run(optimizer, feed_dict = {X: xs, y: ys})
			# sess.run(optimizer, feed_dict = {X: train_imgs, y: train_labels})

			# xs, ys = mnist.test.images, mnist.test.labels
			# c = sess.run(cost, feed_dict = {X: test_imgs, y: test_labels})
			# c = sess.run(cost, feed_dict = {X: xs, y: ys})
			# costs.append(c)

		xs, ys = mnist.test.next_batch(5000)
		print("Cost:", sess.run(cost, feed_dict = {X: xs, y: ys}))
		print("Accuracy:", sess.run(acc, feed_dict = {X: xs, y: ys}))
		print("Time spent for the epoch:", str(time.clock_gettime(0) - t_epoch))

	print("Training is over")
	xs, ys = mnist.test.images, mnist.test.labels
	print("Cost:", sess.run(cost, feed_dict = {X: xs, y: ys}))
	print("Accuracy:", sess.run(acc, feed_dict = {X: xs, y: ys}))
	print("Time spent for the training:", str(time.clock_gettime(0) - t_train))

	# plt.plot(costs)
	# plt.show()
