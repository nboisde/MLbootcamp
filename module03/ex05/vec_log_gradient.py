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

def vec_log_gradient(x, y, theta):
	if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
		raise TypeError("varaibles must be np.arrays")
	xp = np.c_[np.ones(x.shape[0]), x]
	return (1/y.size) * (np.transpose(xp)).dot(logistic_predict_(x, theta) - y)

y1 = np.array([1])
x1 = np.array([4])
theta1 = np.array([[2], [0.5]])
print(vec_log_gradient(x1, y1, theta1))

y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
print(vec_log_gradient(x2, y2, theta2))

y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(vec_log_gradient(x3, y3, theta3))