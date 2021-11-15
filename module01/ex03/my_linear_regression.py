from typing import Type
import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression():
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		if not isinstance(thetas, list):
			raise TypeError("thetas should be a list")
		if len(thetas) != 2 or not isinstance(thetas[0], (int, float)) or not isinstance(thetas[1], (int, float)):
			raise ValueError("Thetas should be a list of two floats")
		if not isinstance(alpha, float):
			raise TypeError("Alpha should be a float")
		if not (alpha <= 1 and alpha >= 0):
			raise ValueError("Alpha should be between 0 and 1")
		if not isinstance(max_iter, int):
			raise TypeError("max_iter should be an int")
		if not max_iter >= 0:
			raise ValueError("Max_iter should be positive.")
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
	
	@staticmethod
	def loss_elem_(y, y_hat):
		if type(y) != type(np.ndarray([])) or type(y_hat) != type(np.ndarray([])):
			return None
		if y.size != y_hat.size:
			return None
		if y.ndim == 1:
			y = y.reshape(y.size, 1)
		if y_hat.ndim == 1:
			y_hat = y_hat.reshape(y_hat.size, 1)
		M = y.size
		tab = []
		x = np.array([])
		for y_val, y_hat_val in np.nditer([y, y_hat]):
			under = []
			tmp = (y_hat_val - y_val)**2
			under.append(tmp)
			tab.append(under)
		res = np.array(tab)
		return res
	
	@staticmethod
	def loss_(y, y_hat):
		if type(y) != type(np.ndarray([])) or type(y_hat) != type(np.ndarray([])):
			return None
		if y.size != y_hat.size:
			return None
		M = y.size
		tab = []
		for yv, yhv in zip(y, y_hat):
			tab.append((yhv - yv))
		vec = np.array(tab)
		vect = np.transpose(vec)
		#print(vect)
		#print(vec)
		return ((1/(2 * M)) * (vect.dot(vec)))[0][0]
		#return (vec.dot(vec))

	def gradient_(self, x, y, th):
		if type(x) != type(np.ndarray([])) or type(y) != type(np.ndarray([])):
			return None
		arrx = np.append([], x)
		arry = np.append([], y)
		nabla = 0
		if arrx.ndim == 1:
			arrx = arrx.reshape(arrx.size, 1)
		xi = np.insert(arrx, 0, 1, axis=1)
		xt = np.transpose(xi)
		M = xt.shape[1]
		nabla = (xt.dot(np.matmul(xi, th) - arry))/M
		return nabla

	def fit_(self, x, y):
		t1 = self.thetas[0] - self.alpha * self.gradient_(x, y, self.thetas)[0]
		t2 = self.thetas[1] - self.alpha * self.gradient_(x, y, self.thetas)[1]
		nthet = np.append([], [t1, t2])
		for i in range(self.max_iter - 1):
			t1 = t1 - self.alpha * self.gradient_(x, y, nthet)[0]	
			t2 = t2 - self.alpha * self.gradient_(x, y, nthet)[1]
			nthet = np.append([], [t1, t2])
		self.thetas = nthet
		return nthet

	def predict_(self, x):
		if type(x) != type(np.ndarray([])):
			return None
		if x.ndim == 1:
			x = x.reshape(x.size, 1)
		x = np.insert(x, 0, 1, axis=1)
		y_hat = x.dot(self.thetas)
		return y_hat

	def plot_with_cost(self, x, y):
		#ybis = theta[0] + theta[1] * xbis
		ones = np.array([])
		for i in y:
			ones = np.insert(ones, 0, 0, axis=0)
		print(ones)
		y_hat = self.predict_(x)
		print(y_hat)
		plt.plot(x, y_hat, 'r')
		plt.scatter(x, y)
		plt.vlines(x, y, y_hat, 'g', '--')
		plt.title("Cost: " + str(format(2 * self.loss_(y, y_hat), '.6f')))
		plt.show()

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
lr1 = MyLinearRegression([2, 0.7])
print(lr1.predict_(x))
print(lr1.loss_elem_(lr1.predict_(x),y))
print(lr1.loss_(lr1.predict_(x),y))

lr2 = MyLinearRegression([1, 1], 5e-5, 1500)
#lr2 = MyLinearRegression([1, 1], 5e-8, 1500000) // same shit, too long, compensate gradient descent with factors of ten,
## it's cheating a little bit because we don't actually know now if we have rights values.
print(lr2.fit_(x, y))
print(lr2.thetas)
print(lr2.predict_(x))
print(MyLinearRegression.loss_elem_(lr2.predict_(x),y))
print(MyLinearRegression.loss_(lr2.predict_(x),y))
lr2.plot_with_cost(x, y)