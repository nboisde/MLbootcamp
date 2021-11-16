import numpy as np

def gradient(x, y, theta):
	if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
		raise TypeError("varaibles must be np.arrays")
	if y.shape[0] != x.shape[0] or x.shape[1] + 1 != theta.shape[0]:
		raise ValueError("impossible to calculate due to error dimentions")
	x = np.c_[np.ones(x.shape[0]), x]
	return ((1/y.size) * np.transpose(x).dot(x.dot(theta) - y))

x = np.array([[ -6, -7, -9], [ 13, -2, 14], [ -7, 14, -1], [-8, -4, 6], [-5, -9, 6], [ 1, -5, 11], [9,-11, 8]])
y = np.array([2, 14, -13, 5, 12, 4, -19])
theta1 = np.array([0, 3, 0.5, -6])
print(gradient(x, y, theta1))
theta2 = np.array([0, 0, 0, 0])
print(gradient(x, y, theta2))