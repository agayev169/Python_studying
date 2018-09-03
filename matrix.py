import random as rand

##############################################################################

class matrix:
	def __init__(self, rows, cols):
		self.__rows = rows
		self.__cols = cols
		self.data = [[0 for x in xrange(cols)] for x in xrange(rows)]

##############################################################################

	def randomize(self):
		for i in range(self.__rows):
			for j in range(self.__cols):
				self.data[i][j] = rand.random() * 2 - 1

##############################################################################

	def add(self, n):
		if type(n) is int:
			for i in range(self.__rows):
				for j in range(self.__cols):
					self.data[i][j] += n
		if isinstance(n, matrix):
			if self.__cols == n.__cols and self.__rows == n.__rows:
				for i in range(self.__rows):
					for j in range(self.__cols):
						self.data[i][j] += n.data[i][j]
			else:
				print "Error in matrix.add() function. The matrices don't match"
				exit()

##############################################################################

	def scalMult(self, n):
		for i in range(self.__rows):
				for j in range(self.__cols):
					self.data[i][j] *= n

##############################################################################

	def hadamardMult(self, m):
		if self.__cols == m.__cols and self.__rows == m.__rows:
			for i in range(self.__rows):
				for j in range(self.__cols):
					self.data[i][j] *= m.data[i][j]
		else:
			print "Error in matrix.mult() function. The matrices don't match"
			exit()

##############################################################################

	@staticmethod
	def mult(a, b):
		if a.__cols == b.__rows:
			m = matrix(a.__rows, b.__cols)
			for i in range(m.__rows):
				for j in range(m.__cols):
					sum = 0
					for k in range(a.__cols):
						sum += a.data[i][k] * b.data[k][j]
					m.data[i][j] = sum
			return m
		else:
			print "Error in matrix.mult() function. The matrices don't match"
			exit()

##############################################################################



##############################################################################

	# @staticmethod 
	# def mult(a, b):
	# 	if a.__cols == b.__rows:
	# 		m = matrix(a.__rows, b.__cols)
	# 		for i in range(m.__rows):
	# 			for j in range(m.__cols):
	# 				sum = 0
	# 				for k in range(a.__cols):
	# 					sum += a.data[i][k] * b.data[k][j]
	# 				m.data[i][j] = sum
	# 		return m
	# 	else:
	# 		print "Error in matrix.mult() function. The matrices don't match"
	# 		exit()

##############################################################################

	# def subtract(self, m):
	# 	if isinstance(n, matrix):
	# 		if self.__cols == n.__cols and self.__rows == n.__rows:
	# 			for i in range(self.__rows):
	# 				for j in range(self.__cols):
	# 					self.data[i][j] -= n.data[i][j]
	# 		else:
	# 			print "Error in matrix.subtract() function. The matrices don't match"
	# 			exit()

##############################################################################

	@staticmethod
	def subtract(m, n):
		if isinstance(n, matrix) and isinstance(m, matrix):
			if m.__cols == n.__cols and m.__rows == n.__rows:
				new = matrix(m.__rows, m.__cols)
				for i in range(new.__rows):
					for j in range(new.__cols):
						new.data[i][j] = m.data[i][j] - n.data[i][j]
			else:
				print "Error in matrix.subtract() function. The matrices don't match"
				exit()
		return new

##############################################################################

	def getData(self):
		return self.data

##############################################################################

	def setData(self, data, x, y):
		self.data[x][y] = data

##############################################################################

	@staticmethod
	def transpose(m):
		new = matrix(m.__cols, m.__rows)
		for i in range(m.__rows):
			for j in range(m.__cols):
				new.data[j][i] = m.data[i][j]
		return new


##############################################################################

	@staticmethod
	def fromArray(arr):
		if (isinstance(arr, list)):
			m = matrix(len(arr), 1)
			for i in range(len(arr)):
				m.data[i][0] = arr[i]
			return m
		else:
			print "Error in matrix.fromArray() function. The given argument is not a list"

##############################################################################

	@staticmethod
	def toArray(m):
		if (isinstance(m, matrix)):
			arr = []
			for i in range(len(m.data)):
				arr.append(m.data[i][0])
			return arr
		else:
			print "Error in matrix.fromArray() function. The given argument is not a matrix"

##############################################################################

	def __str__(self):
		return str(self.data)

##############################################################################
