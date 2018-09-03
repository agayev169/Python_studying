from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument(
	'--data_dir',
	type=str,
	default='/tmp/tensorflow/mnist/input_data',
help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()




mnist = input_data.read_data_sets(FLAGS.data_dir)

n_input = 784
n_hidden = 100
n_output = 10
learning_rate = 0.01
epochs = 10000

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Weights and Biases
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
B1 = tf.Variable(tf.zeros([n_hidden]))

W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
B2 = tf.Variable(tf.zeros([n_output]))

H = tf.sigmoid(tf.matmul(X, W1) + B1)
Ows = tf.matmul(H, W2) + B2
O = tf.sigmoid(tf.matmul(H, W2) + B2)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Ows, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

costs = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(epochs):
		batch_xs, batch_ys = mnist.train.next_batch(100)

		sess.run(optimizer, feed_dict = {X: batch_xs, Y: batch_ys})

		c = sess.run(cost, feed_dict = {X: batch_xs, Y: batch_ys})
		costs.append(c)

		# if epoch % 100 == 0:
		print (sess.run(cost, feed_dict = {X: batch_xs, Y: batch_ys}))

	answer = tf.equal(tf.floor(O + 0.5), Y)
	accuracy = tf.reduce_mean(tf.cast(answer, "float"))

	print (sess.run(O, feed_dict = {X: batch_xs, Y: batch_ys}))
	print (np.round(sess.run(O, feed_dict = {X: batch_xs, Y: batch_ys})))

	print ("Training is over")

	plt.plot(costs)
	plt.show()