import numpy as np
import math

def sigmoid_(x):
	if not isinstance(x, np.ndarray):
		raise TypeError("x must be a np array")
	sgm = lambda t: 1 / (1 + math.exp(-t))
	if x.ndim == 0:
		sigmoid = np.array(sgm(x))
		ret = np.append([], sigmoid)
	elif x.ndim == 1:
		sigmoid = np.array([sgm(xi) for xi in x])
		ret = sigmoid.reshape(-1, 1)
	else:
		sigmoid = []
		for val in x:
			for elem in val:
				tmp = np.array(sgm(elem))
				sigmoid.append(tmp)
		sigmoid = np.array(sigmoid)
		ret = sigmoid.reshape(-1, 1)
	return ret

def logistic_predict_(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		raise TypeError("should be numpy arrays")
	th = theta.reshape((len(theta), 1))
	x = np.c_[np.ones(x.shape[0]), x]
	if th.shape[0] != x.shape[1]:
		raise ValueError("multiplictation impossible...")
	tmp = x.dot(theta)
	return sigmoid_(tmp)

# cross entropy
def log_loss_(y, y_hat, eps=1e-15):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		raise TypeError("y and y_hat should be numpy arrays")
	if y.ndim != y_hat.ndim:
		raise ValueError("y and y_hat must have same dim")
	for i in range(len(y_hat)):
		if y[i] == 0:
			y[i] = eps
		if y_hat[i] == 0:
			y_hat[i] = eps
	print(y)
	print(y_hat)

y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
y_hat2 = logistic_predict_(x2, theta2)
print(log_loss_(y2, y_hat2))