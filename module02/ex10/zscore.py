from typing import Type
import numpy as np

def zscore(X):
	if not isinstance(X, np.ndarray):
		raise TypeError("Should be a np array")
	if X.ndim != 1 and X.ndim != 2:
		raise ValueError("Should be a vector")
	return (X - np.mean(X)) / np.std(X)
