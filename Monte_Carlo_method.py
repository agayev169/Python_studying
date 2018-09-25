import random
import math
# import time

def f(x):
	return x * x

x_min = 2
x_max = 50


y_min = f(x_min)
y_max = f(x_max)

total = 500000
under = 0

for _ in range(total):
	x_rand = random.uniform(x_min, x_max)
	y_rand = random.uniform(y_min, y_max)
	y = f(x_rand)
	if y_rand <= y:
		under += 1

print(str((under / total) * (x_max - x_min) * (y_max - y_min) + (x_max - x_min)))