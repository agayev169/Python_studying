import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

# Data
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0]   , [1]   , [1]   , [0]])

# Hyperparameters
# n_input = 2
# n_hidden = 4
# n_output = 1
# learning_rate = 0.1
# epochs = 1000

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4, activation = "sigmoid"))
model.add(tf.keras.layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", 
	loss = "sparce_cathegorical_crossentropy", 
	metrics =["accuracy"])

model.fit(x_data, y_data, epochs = 10, batch_size = 1)


# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)

# # Weights and Biases
# W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
# B1 = tf.Variable(tf.zeros([n_hidden]))

# W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
# B2 = tf.Variable(tf.zeros([n_output]))

# H = tf.sigmoid(tf.matmul(X, W1) + B1)
# Ows = tf.matmul(H, W2) + B2
# O = tf.sigmoid(tf.matmul(H, W2) + B2)

# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Ows, labels = Y))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# costs = []

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)

# 	for epoch in xrange(epochs):
# 		sess.run(optimizer, feed_dict = {X: x_data, Y: y_data})

# 		c = sess.run(cost, feed_dict = {X: x_data, Y: y_data})
# 		costs.append(c)

# 		if epoch % 100 == 0:
# 			print sess.run(cost, feed_dict = {X: x_data, Y: y_data})

# 	answer = tf.equal(tf.floor(O + 0.5), Y)
# 	accuracy = tf.reduce_mean(tf.cast(answer, "float"))

# 	print sess.run(O, feed_dict = {X: x_data, Y: y_data})
# 	print np.round(sess.run(O, feed_dict = {X: x_data, Y: y_data}))

# 	print "Training is over"

# 	plt.plot(costs)
# 	plt.show()