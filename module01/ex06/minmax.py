import numpy as np

def minmax(x):
	if not isinstance(x, np.ndarray):
		raise TypeError("Should be a np array")
	if x.ndim != 1:
		raise ValueError("Should be a vector")
	return (x - np.min(x)) / (np.max(x) - np.min(x))

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(minmax(X))

Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(minmax(Y))