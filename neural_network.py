from matrix import *
import math
import random as rand

##############################################################################

class neural_network:
	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate = 0.01):
		self.__input_nodes = input_nodes
		self.__hidden_nodes = hidden_nodes
		self.__output_nodes = output_nodes
		self.__learning_rate = learning_rate

		self.weights_ih = matrix(hidden_nodes, input_nodes)
		self.weights_ho = matrix(output_nodes, hidden_nodes)
		self.weights_ih.randomize()
		self.weights_ho.randomize()

		self.__bias_h = matrix(hidden_nodes, 1)
		self.__bias_o = matrix(output_nodes, 1)
		self.__bias_h.randomize()
		self.__bias_o.randomize()

##############################################################################

	# def __init__(self, nn):
	# 	if isinstance(nn, neural_network):
	# 		self.__input_nodes = nn.__input_nodes
	# 		self.__hidden_nodes = nn.__hidden_nodes
	# 		self.__output_nodes = nn.__output_nodes
	# 		self.__learning_rate = nn.__learning_rate

	# 		self.weights_ih = nn.weights_ih
	# 		self.weights_ho = nn.weights_ho

	# 		self.__bias_h = nn.__bias_h
	# 		self.__bias_o = nn.__bias_o

##############################################################################

	def guess(self, inputs_arr):
		inputs = matrix.fromArray(inputs_arr)
		hidden = matrix.mult(self.weights_ih, inputs)
		hidden.add(self.__bias_h)

		neural_network.activation_function(hidden, neural_network.sigmoid)

		outputs = matrix.mult(self.weights_ho, hidden)
		outputs.add(self.__bias_o)
		
		neural_network.activation_function(outputs, neural_network.sigmoid)
		# print outputs
		# print matrix.toArray(outputs)
		return matrix.toArray(outputs)

##############################################################################

	def train(self, inputs_arr, targets_arr):
		inputs = matrix.fromArray(inputs_arr)
		hidden = matrix.mult(self.weights_ih, inputs)
		hidden.add(self.__bias_h)

		neural_network.activation_function(hidden, neural_network.sigmoid)

		outputs = matrix.mult(self.weights_ho, hidden)
		outputs.add(self.__bias_o)
		
		neural_network.activation_function(outputs, neural_network.sigmoid)

		targets = matrix.fromArray(targets_arr)

		output_errors = matrix.subtract(targets, outputs)

		output_gradients = neural_network.activation_function(outputs, neural_network.dsigmoid)
		output_gradients.hadamardMult(output_errors)
		output_gradients.scalMult(self.__learning_rate)

		hidden_T = matrix.transpose(hidden)
		weights_ho_deltas = matrix.mult(output_gradients, hidden_T)

		self.weights_ho.add(weights_ho_deltas)
		self.__bias_o.add(output_gradients)

		weights_ho_T = matrix.transpose(self.weights_ho)
		hidden_errors = matrix.mult(weights_ho_T, output_errors)

		hidden_gradients = neural_network.activation_function(hidden, neural_network.dsigmoid)
		hidden_gradients.hadamardMult(hidden_errors)
		hidden_gradients.scalMult(self.__learning_rate)

		inputs_T = matrix.transpose(inputs)
		weights_ih_deltas = matrix.mult(hidden_gradients, inputs_T)

		self.weights_ih.add(weights_ih_deltas)
		self.__bias_h.add(hidden_gradients)

##############################################################################

	@staticmethod
	def activation_function(m, func):
		for i in range(len(m.data)):
			for j in range(len(m.data[0])):
				m.data[i][j] = func(m.data[i][j])
		return m

##############################################################################
	
	@staticmethod
	def sigmoid(x):
		return (1 / (1 + math.exp(-x)))

##############################################################################

	@staticmethod
	def dsigmoid(x):
		return (x * (1 - x))

##############################################################################

	def __str__(self):
		return str(self.weights_ih) + "\n" + str(self.weights_ho)

##############################################################################

def maxI(l):
	max = 0
	for x in xrange(1,len(l)):
		if l[x] > l [max]:
			max = x
	return max

##############################################################################



nn = neural_network(2, 2, 2, 0.1)

for x in xrange(75000):
	inputs_arr = [0, 0]
	inputs_arr[0] = int(rand.random() * 2)
	inputs_arr[1] = int(rand.random() * 2)
	if (inputs_arr[0] == 1 and inputs_arr[1] == 1) or (inputs_arr[0] == 0 and inputs_arr[1] == 0):
		targets_arr = [0, 1]
	else:
		targets_arr = [1, 0]
	# print str(inputs_arr) + "\t\t" + str(targets_arr)
	nn.train(inputs_arr, targets_arr)

precision = 0
for x in xrange(1000):
	inputs_arr = [0, 0]
	inputs_arr[0] = int(rand.random() * 2)
	inputs_arr[1] = int(rand.random() * 2)
	if (inputs_arr[0] == 1 and inputs_arr[1] == 1) or (inputs_arr[0] == 0 and inputs_arr[1] == 0):
		targets_arr = [0, 1]
	else:
		targets_arr = [1, 0]
	# if x % 100 == 0:
	# 	print(str(maxI(nn.guess(inputs_arr))) + " " + str(maxI(targets_arr)))
	if maxI(nn.guess(inputs_arr)) == maxI(targets_arr):
		# print "intersection"
		precision += 1

print str(precision / 1000.0)