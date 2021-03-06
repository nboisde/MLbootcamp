import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

## On calcule la probabilit√© d'etre proche de 1 dans un modele de regression logistique !
##
class MyLogisticRegression():
	def __init__(self, theta, alpha=0.001, max_iter=1000):
		if not isinstance(alpha, (float, int)):
			raise TypeError("alpha and max_iter should be numbers")
		if alpha < 0 or alpha > 1:
			raise ValueError("alpha should be between 0 and 1")
		if not isinstance(max_iter, int):
			raise TypeError("max_iter should be an int")
		if max_iter < 0:
			raise ValueError("iteration number must be positive")
		self.theta = np.array(theta)
		self.alpha = alpha
		self.max_iter = max_iter

	def sigmoid_(self, x):
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

	def predict_(self, x):
		if not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray):
			raise TypeError("should be numpy arrays")
		th = self.theta.reshape((len(self.theta), 1))
		x = np.c_[np.ones(x.shape[0]), x]
		if th.shape[0] != x.shape[1]:
			raise ValueError("multiplictation impossible...")
		tmp = x.dot(self.theta)
		return self.sigmoid_(tmp)

	def vec_log_gradient(self, x, y):
		if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(self.theta, np.ndarray):
			raise TypeError("varaibles must be np.arrays")
		xp = np.c_[np.ones(x.shape[0]), x]
		return (1/y.size) * (np.transpose(xp)).dot(self.predict_(x) - y)

	def loss_(self, x, y, eps=1e-15):
		if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
			raise TypeError("y and y_hat should be numpy arrays")
		y = y.flatten()
		y_hat = self.predict_(x)
		y_hat = y_hat.flatten()	
		return - 1 / y.shape[0] * (np.dot(y, np.log(y_hat + eps)) + np.dot(np.ones(y.shape[0]) - y, np.log(np.ones(y.shape[0]) - (y_hat - eps))))

	def fit_(self, x, y):
		if not isinstance(self.alpha, (float, int)):
			raise TypeError("alpha and max_iter should be numbers")
		if self.alpha < 0 or self.alpha > 1:
			raise ValueError("alpha should be between 0 and 1")
		if not isinstance(self.max_iter, int):
			raise TypeError("max_iter should be an int")
		if self.max_iter < 0:
			raise ValueError("iteration number must be positive")
		if x.ndim != 1:
			self.theta = self.theta.reshape((len(self.theta), 1))
		for i in range(self.max_iter):
			self.theta = self.theta - self.alpha * self.vec_log_gradient(x, y)
		return self.theta



