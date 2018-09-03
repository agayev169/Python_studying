import math as m

class Point:
	def __init__(self, x = 0, y = 0):
		self.__x = x
		self.__y = y

	def setX(self, x):
		self.__x = x

	def setY(self, y):
		self.__y = y

	def getX(self):
		return self.__x

	def getY(self):
		return self.__y

	def __str__(self):
		return "x: " + str(self.__x) + " y: " + str(self.__y)

	def dist(self):
		return m.sqrt(self.__x**2 + self.__y**2)

	@staticmethod
	def barycenter(l):
		x, y = 0, 0
		for i in l:
			x += i.getX()
			y += i.getY()
		
		x /= len(l)
		y /= len(l)

		return Point(x, y)

p = Point()
l = [Point(3, 4), Point(1, 1)]
print(str(Point.barycenter(l)))