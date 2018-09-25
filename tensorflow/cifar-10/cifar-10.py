import tensorflow as tf
import numpy as np
import time
import random

'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def prepare_data(data):
	x_train_tmp, y_train_tmp = data[b'data'], data[b'labels']
	x_train = []
	y_train = []
	for i in range(len(y_train_tmp)):
		y_train.append([])
		y_train[i] = [0 for _ in range(10)]
		y_train[i][y_train_tmp[i]] = 1

	for i in range(len(x_train_tmp)):
		x_train.append([])
		for j in range(1024):
			th = int(x_train_tmp[i][j]) + int(x_train_tmp[i][j+1024]) + int(x_train_tmp[i][j+1024])
			th /= 765.0
			x_train[i].append(th)

	x_train, y_train = np.array(x_train), np.array(y_train)
	return x_train, y_train

print("Unpickling file #1/5")
file = 'cifar-10-batches-py/data_batch_1'
data = unpickle(file)
x_train, y_train = prepare_data(data)


for i in range(4):
	print("Unpickling file #%i/5" %(i + 2))
	file = 'cifar-10-batches-py/data_batch_' + str(i + 2)
	data = unpickle(file)

	x_tmp, y_tmp = prepare_data(data)
	x_train, y_train = np.append(x_train, x_tmp, axis = 0), np.append(y_train, y_tmp, axis = 0)

file = 'cifar-10-batches-py/test_batch'
data = unpickle(file)
x_test, y_test = prepare_data(data)

file = 'x_train_1.npy'
np.save(file, x_train)
file = 'y_train_1.npy'
np.save(file, y_train)
file = 'x_test_1.npy'
np.save(file, x_test)
file = 'y_test_1.npy'
np.save(file, y_test)
'''
# file = open("train_data.py", "w")

# file.write("import numpy as np\n")
# np.set_printoptions(threshold=np.inf)
# file.write("x_train = np.array(")
# file.write(np.array2string(x_train, separator=','))
# file.write(")\ny_train = np.array(")
# file.write(np.array2string(y_train, separator=','))
# file.write(")\n")

# file.close()
# print("Data is prepared")
# quit()


t = time.clock_gettime(0)
x_train = np.load('x_train_3.npy')
y_train = np.load('y_train_3.npy')
x_test = np.load('x_test_3.npy')
y_test = np.load('y_test_3.npy')
print("Time spent to load data:", time.clock_gettime(0) - t)

n_classes = 10
lr = 0.001

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
'''
def getActivation(layer, img):
	units = sess.run(layer, feed_dict={x: np.reshape(img, [1, 1024], order = 'F')})
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
'''

def convolutional_neural_network(x):
	weights = {'conv1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
			   'conv2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
			   'conv3': tf.Variable(tf.random_normal([3, 3, 32, 48])),
			   'conv4': tf.Variable(tf.random_normal([3, 3, 48, 48])),
			   'fc1': tf.Variable(tf.random_normal([8*8*48, 256])),
			   'fc2': tf.Variable(tf.random_normal([256, 100])),
			   'fc3': tf.Variable(tf.random_normal([100, 100])),
			   'out': tf.Variable(tf.random_normal([100, n_classes]))}

	biases = {'conv1': tf.Variable(tf.random_normal([32])),
			  'conv2': tf.Variable(tf.random_normal([32])),
			  'conv3': tf.Variable(tf.random_normal([48])),
			  'conv4': tf.Variable(tf.random_normal([48])),
			  'fc1': tf.Variable(tf.random_normal([256])),
			  'fc2': tf.Variable(tf.random_normal([100])),
			  'fc3': tf.Variable(tf.random_normal([100])),
			  'out': tf.Variable(tf.random_normal([n_classes]))}

	x = tf.reshape(x, shape=[-1, 32, 32, 3])

	conv1 = tf.nn.relu(conv2d(x, weights['conv1']) + biases['conv1'])
	conv1 = tf.nn.lrn(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['conv2']) + biases['conv2'])
	conv2 = maxpool2d(conv2)
	conv2 = tf.nn.lrn(conv2)

	conv3 = tf.nn.relu(conv2d(conv2, weights['conv3']) + biases['conv3'])
	conv3 = tf.nn.lrn(conv3)

	conv4 = tf.nn.relu(conv2d(conv3, weights['conv4']) + biases['conv4'])
	conv4 = maxpool2d(conv4)
	conv4 = tf.nn.lrn(conv4)

	fc1 = tf.reshape(conv4, [-1, 8*8*48])
	fc1 = tf.nn.relu(tf.matmul(fc1, weights['fc1']) + biases['fc1'])

	fc2 = tf.nn.relu(tf.matmul(fc1, weights['fc2']) + biases['fc2'])

	fc3 = tf.nn.relu(tf.matmul(fc2, weights['fc3']) + biases['fc3'])

	return tf.matmul(fc3, weights['out']) + biases['out']

train = convolutional_neural_network(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = train, labels = y))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(train), 1), tf.argmax(y, 1)), tf.float32))

saver = tf.train.Saver(max_to_keep = None)

def next_batch(x, y, batch_size, offset):
	return x[offset:offset + batch_size], y[offset:offset + batch_size]


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	batch_size = 128
	epochs = 15
	acc_max = 0
	acc_max_index = 0
	cost_min = 10000000
	cost_min_index = 0
	save_count = 0
	t_train = time.clock_gettime(0)
	for epoch in range(epochs):
		t_epoch = time.clock_gettime(0)
		for n in range(int(len(x_train) / batch_size)):
			xs, ys = next_batch(x_train, y_train, batch_size, n * batch_size)
			sess.run(optimizer, feed_dict = {x: xs, y: ys})
			index = random.randint(0, len(x_test) - 1001)
			xs, ys = x_test[index:index + 1000], y_test[index:index + 1000];
			c, acc_cur = sess.run([cost, acc], feed_dict = {x: xs, y: ys})
			if acc_cur > acc_max or c < cost_min:
				save_path = saver.save(sess, "models/cnn_1/model" + str(save_count) + ".ckpt")
				print("Accuracy:", acc_cur)
				print("Cost:", c)
				if acc_cur > acc_max:
					acc_max_index = save_count
					acc_max = acc_cur
					print("Best accuracy so far")
				if c < cost_min:
					cost_min_index = save_count
					cost_min = c
					print("Best cost so far")
				save_count += 1
				print("Model saved in path: %s\n" % save_path)
		print("Epoch #%i/%i" %(epoch + 1, epochs))
		print("Cost:", c)
		index = random.randint(0, len(x_test) - 1001)
		xs, ys = x_test[index:index + 1000], y_test[index:index + 1000];
		accuracy = sess.run(acc, feed_dict = {x: xs, y: ys})
		print("Accuracy:", accuracy)
		print("Time spent for the epoch:", time.clock_gettime(0) - t_epoch)
		print("Epoch has end at:", time.localtime())
		file = open("models/cnn_1/info", 'a')
		file.write("cost_min = %f at %i, acc_max = %f at %i\n" 
			%(cost_min, cost_min_index, acc_max, acc_max_index))
		file.close()

	print("Training completed")
	print("Time spent for the training:", time.clock_gettime(0) - t_train)