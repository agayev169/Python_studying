import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import math

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_input = 784
n_classes = 10
lr = 0.01


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

# keep_rate = 0.8
# keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def getActivation(layer, img):
	units = sess.run(layer, feed_dict={x: np.reshape(img, [1, 784], order = 'F')})
	plotNNFilter(units)

def plotNNFilter(units):
	filters = units.shape[3]
	plt.figure(1, figsize = (20, 20))
	n_cols = 6
	n_rows = math.ceil(filters / n_cols)
	for i in range(filters):
		plt.subplot(n_rows, n_cols, i + 1)
		plt.title("Filter " + str(i + 1))
		plt.imshow(units[0, :, :, i], interpolation = 'nearest', cmap = 'gray')
	plt.show()


weights = {'conv1': tf.Variable(tf.random_normal([3, 3, 1, 4])),
		   'conv2': tf.Variable(tf.random_normal([3, 3, 4, 16])),
		   'fc1': tf.Variable(tf.random_normal([7*7*16, 256])),
		   'fc2': tf.Variable(tf.random_normal([256, 100])),
		   'out': tf.Variable(tf.random_normal([100, n_classes]))}

biases = {'conv1': tf.Variable(tf.random_normal([4])),
		  'conv2': tf.Variable(tf.random_normal([16])),
		  'fc1': tf.Variable(tf.random_normal([256])),
		  'fc2': tf.Variable(tf.random_normal([100])),
		  'out': tf.Variable(tf.random_normal([n_classes]))}

x_in = tf.reshape(x, shape=[-1, 28, 28, 1])

conv1 = tf.sigmoid(conv2d(x_in, weights['conv1']) + biases['conv1'])
pool1 = maxpool2d(conv1)

conv2 = tf.sigmoid(conv2d(pool1, weights['conv2']) + biases['conv2'])
pool2 = maxpool2d(conv2)

fc1 = tf.reshape(pool2, [-1, 7*7*16])
fc1 = tf.sigmoid(tf.matmul(fc1, weights['fc1']) + biases['fc1'])

fc2 = tf.sigmoid(tf.matmul(fc1, weights['fc2']) + biases['fc2'])

train = tf.matmul(fc2, weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = train, labels = y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train, 1), tf.argmax(y, 1)), tf.float32))
pred = tf.argmax(train, 1)
pred_bool = tf.equal(pred, tf.argmax(y, 1))

costs = []
accs  = []

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep = None)

with tf.Session() as sess:
	sess.run(init)

	# for i in range(1, 4):
	# saver.restore(sess, "models/model" + str(i) + ".ckpt")

	saver.restore(sess, "models/cnn_4/model91.ckpt")
	print("Model restored")
	# acc_sum = 0
	# for _ in range(int(len(mnist.test.images) / 1000)):
	# 	xs, ys = mnist.test.next_batch(1000)
	# 	acc_in = sess.run(acc, feed_dict = {x: xs, y: ys})
	# 	acc_sum += acc_in

	# print(acc_sum / (len(mnist.test.images) / 1000))

	# for i in range(20):
	# 	getActivation(conv1, mnist.test.images[i])

	pred_wrong = []
	predicted = []
	value = []
	for i in range(1000):
		x_input, y_input = mnist.test.images[i], mnist.test.labels[i]
		if not sess.run(pred_bool, 
				feed_dict = {x: np.reshape(x_input, [1, 784], order = 'F'), 
				y: np.reshape(y_input, [1, 10], order = 'F')}):
			pred_wrong.append(i)
			value.append(sess.run(tf.argmax(y, 1), 
				feed_dict = {y: np.reshape(y_input, [1, 10], order = 'F')}))
			predicted.append(sess.run(pred, 
				feed_dict = {x: np.reshape(x_input, [1, 784], order = 'F')})) 

	print(len(pred_wrong))
	for i in range(len(pred_wrong)):
		plt.imshow(np.reshape(mnist.test.images[pred_wrong[i]], [28, 28]))
		print("Real value:", value[i], "Predicted as:", predicted[i])
		plt.show()
	# print(sess.run(train, feed_dict = {x: xs, y: ys}))
	# print(sess.run(pred, feed_dict = {x: xs, y: ys}))
	# print(ys)
	# print(sess.run(tf.argmax(train, 1), feed_dict = {x: xs, y: ys}))
	# print(sess.run(tf.argmax(ys, 1)))
	# print(sess.run(acc, feed_dict = {x: xs, y: ys}))

	# epochs = 20
	# batch_size = 128

	# cost_min = 0.2
	# cost_min_index = 0
	# acc_max = 0.50
	# acc_max_index = 0
	# # acc_avg = 0.98
	# save_count = 1
	# for epoch in range(epochs):
	# 	epoch_time = time.clock_gettime(0);
	# 	epoch_max_acc = 0
	# 	for i in range(int(len(mnist.train.images) / batch_size)):
	# 		xs, ys = mnist.train.next_batch(batch_size)
	# 		sess.run(optimizer, feed_dict = {x: xs, y: ys})
	# 		c = sess.run(cost, feed_dict = {x: xs, y: ys})
	# 		costs.append(c)
	# 		xs, ys = mnist.test.next_batch(3000)
	# 		accur = sess.run(acc, feed_dict = {x: xs, y: ys})
	# 		accs.append(accur)

	# 		if (accur > epoch_max_acc):
	# 			epoch_max_acc = accur
	# 		if c < cost_min or accur > acc_max:
	# 			save_path = saver.save(sess, "models/cnn_4/model" + str(save_count) + ".ckpt")
	# 			print("Cost:", c)
	# 			print("Accuracy:", accur)
	# 			save_count += 1
	# 			if c < cost_min:
	# 				print("Best cost so far")
	# 				cost_min = c
	# 				cost_min_index = save_count - 1
	# 			if acc_max < accur:
	# 				print("Best accuracy so far")
	# 				acc_max = accur
	# 				acc_max_index = save_count - 1
	# 			print("Model saved in path: %s\n" % save_path)
			

	# 	print("Results after the epoch #" + str(epoch + 1) + "/" + str(epochs))
	# 	print("Max average in the epoch:", epoch_max_acc)
	# 	print("cost_min = %f at %i, acc_max = %f at %i" 
	# 		%(cost_min, cost_min_index, acc_max, acc_max_index))
	# 	print("Time spent for the epoch:", time.clock_gettime(0) - epoch_time, "\n\n")
	# 	file = open("models/cnn_4/info", 'a')
	# 	file.write("cost_min = %f at %i, acc_max = %f at %i\n" 
	# 		%(cost_min, cost_min_index, acc_max, acc_max_index))
	# 	file.close()

	# print("Training is over")

	# plt.plot(costs, label = "Cost")
	# plt.plot(accs, label = "Accuracy")
	# plt.legend()
	# plt.show()
