import numpy as np

def add_intercept(x):
	if type(x) != type(np.ndarray([])):
		return None
	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	x = np.insert(x, 0, 1, axis=1)
	return x

x = np.arange(1,6)
print(add_intercept(x))
y = np.arange(1,10).reshape((3,3))
print(add_intercept(y))