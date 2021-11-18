import numpy as np
import math
import copy

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
	if y_hat.ndim == 1:
		y_hat = y_hat.reshape(-1, 1)
	if y.ndim == 1:
		y = y.reshape(-1, 1)
	y2 = []
	y_hat2 = []
	for yv, yhv in zip(y, y_hat):
		tmpy = []
		tmpyh = []
		if yv == 0:
			tmpy.append(eps)
		else:
			tmpy.append(yv[0])
		if yhv == 0:
			tmpyh.append(eps)
		else:
			tmpyh.append(yhv[0])
		y2.append(tmpy)
		y_hat2.append(tmpyh)
	sum = 0
	print(y2)
	print(y_hat2)
	for yv, yhv in zip(y2, y_hat2):
		tmp = (yv[0] * math.log(yhv[0])) + ((1 - yv[0]) * math.log(1 - yhv[0]))
		sum += tmp
	#return ((-1) / y.size) * sum
	return - 1 / y.shape[0] * (np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))
	

y1 = np.array([1])
x1 = np.array([4])
theta1 = np.array([[2], [0.5]])
y_hat1 = logistic_predict_(x1, theta1)
print(log_loss_(y1, y_hat1))

y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
y_hat2 = logistic_predict_(x2, theta2)
print(log_loss_(y2, y_hat2))

y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
y_hat3 = logistic_predict_(x3, theta3)
print(log_loss_(y3, y_hat3))