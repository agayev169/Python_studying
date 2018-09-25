import random
import matplotlib.pyplot as plt

n_epochs = 10000
n_tosses = 2000
tosses = [0 for _ in range(n_tosses + 1)]

for _ in range(n_epochs):
	n = 0
	for _ in range(n_tosses):
		if random.random() > 0.5:
			n +=1
	tosses[n] += 1

xs = [x for x in range(n_tosses + 1)]
ys = [tosses[x] for x in range(n_tosses + 1)]

plt.scatter(xs, ys, s = 1)
plt.show()