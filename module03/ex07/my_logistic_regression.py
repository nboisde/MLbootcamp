import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

## On calcule la probabilité d'etre proche de 1 dans un modele de regression logistique !
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

	def plot_1D(self, x, y, y_hat, xax="x axis", yax="y axis", yleg="data", yhleg="predicted", coly='blue', colyh='lightblue'):
		plt.xlabel(xax)
		plt.ylabel(yax)
		plt.scatter(x, y, label=yleg, color=coly)
		plt.scatter(x, y_hat, label=yhleg, color=colyh)
		plt.legend()
		plt.grid()
		plt.show()
	
	def plot_3D(self, x, y, y_hat):
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		for i, val in enumerate(y):
			if val == 1.0:
				ax.scatter(x[i][0], x[i][1], x[i][2], color='blue')
			else:
				ax.scatter(x[i][0], x[i][1], x[i][2], color='green')
		for i, val in enumerate(y_hat):
			if val >= 0.5:
				ax.scatter(x[i][0], x[i][1], x[i][2], color='red', marker='.')
			else:
				ax.scatter(x[i][0], x[i][1], x[i][2], color='yellow', marker='.')
		
		plt.show()

	def appartenance_proportion(self, y):
		if not isinstance(y, np.ndarray):
			raise TypeError("appartenance proportion: y should be a np array")
		ones = 0
		zeros = 0
		for val in y:
			if val >= 0.5:
				ones += 1
			else:
				zeros += 1
		l = y.shape[0]
		return ones / l
