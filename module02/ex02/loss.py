from typing import Type
import numpy as np

def loss_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		raise TypeError("y and y_hat should be numpy arrays")
	if y.ndim != y_hat.ndim:
		raise ValueError("y and y_hat must have same dim")
	return (1/(2 * y.size)) * (y - y_hat).dot((y - y_hat))

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(loss_(X, Y))
print(loss_(X, X))
