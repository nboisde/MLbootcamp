import numpy as np

def minmax(x):
	if not isinstance(x, np.ndarray):
		raise TypeError("Should be a np array")
	if x.ndim != 1 and x.ndim != 2:
		raise ValueError("Should be a vector")
	return (x - np.min(x)) / (np.max(x) - np.min(x))
