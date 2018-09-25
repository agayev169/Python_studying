import matplotlib.pyplot as plt
import numpy as np

file = 'x_test_1.npy'
x_test = np.load(file)
file = 'y_test_3.npy'
y_test = np.load(file)

plt.imshow(x_test[11].reshape([32, 32]), cmap = 'gray')
plt.show()