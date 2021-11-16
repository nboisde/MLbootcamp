from typing import Type
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class MyLinearRegression():
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		if not isinstance(thetas, (list, np.ndarray)):
			raise TypeError("thetas should be a list")
		if isinstance(thetas, np.ndarray):
			if thetas.ndim == 2 and thetas.shape == (2, 1):
				thetas = [thetas[0][0], thetas[1][0]]
		#thetas = [thetas[0][0], thetas[1][0]]
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
		print(x)
		print(self.thetas)
		return y_hat

	def plot_prediction_(self, x, y, xleg='x', yleg='y'):
		#ybis = theta[0] + theta[1] * xbis
		ones = np.array([])
		for i in y:
			ones = np.insert(ones, 0, 0, axis=0)
		y_hat = self.predict_(x)
		plt.xlabel(xleg)
		plt.ylabel(yleg)
		plt.grid()
		x2 = x.flatten()
		y_hat2 = y_hat.flatten()
		y2 = y.flatten()
		plt.plot(x2, y_hat2, '--g')
		plt.scatter(x2, y2, color='lightblue')
		plt.scatter(x2, y_hat2, marker='x', color='lightgreen')
		plt.title("Cost: " + str(format(2 * self.loss_(y, y_hat), '.6f')))
		plt.show()
	
	def mse_(self, y, y_hat):
		return (2 * self.loss_(y, y_hat))

	def plot_loss_(self, x, y):
		thetas_1 = np.linspace(-15, 15, 40)
		t01 = -100
		t02 = 200
		t03 = 89
		def loss_curve(t0, pcolor='b'):
			loss_values =[]
			for val in thetas_1:
				y_hat = self.predict_2(x, val, t0)
				loss_values.append((1/(2 * y.size)) * self.loss_2(y, y_hat))
				#loss_values.append(self.r2score_(y, y_hat))
			plt.plot(thetas_1, loss_values, color=pcolor)
		loss_curve(t01)
		loss_curve(t02, pcolor='r')
		loss_curve(t03, pcolor='g')
		plt.show()
	
	@staticmethod
	def loss_2(y, y_hat):
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
		return ((1/(2 * M)) * (vect.dot(vec)))
	
	def r2score_(self, y, y_hat):
		if type(y) != type(np.ndarray([])) or type(y_hat) != type(np.ndarray([])):
			return None
		if y.size != y_hat.size:
			return None
		ym = 0
		for v in y:
			ym += (v - np.mean(y))**2
		M = y.size
		return 1 - ((M * ((1 / (2 * M)) * self.loss_2(y, y_hat))) / ym)

	def predict_2(self, x, t1, t0):
		if type(x) != type(np.ndarray([])):
			return None
		if x.ndim == 1:
			x = x.reshape(x.size, 1)
		x = np.insert(x, 0, 1, axis=1)
		y_hat = x.dot(np.array([t0, t1]))
		return y_hat
		


#data = pd.read_csv('are_blue_pills_magics.csv')
#Xpill = np.array(data['Micrograms']).reshape(-1,1)
#Yscore = np.array(data['Score']).reshape(-1,1)
#
#thetas = np.array([[89.0], [-8]])
#linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
#Y_model1 = linear_model1.predict_(Xpill)
#xleg = "Quantity of blue pills (in microgram)"
#yleg = "Space driving score"
##linear_model1.plot_prediction_(Xpill, Yscore, xleg, yleg)
#linear_model1.fit_(Xpill, Yscore)
#print(linear_model1.thetas)
##linear_model1.plot_prediction_(Xpill, Yscore, xleg, yleg)
#linear_model1.plot_loss_(Yscore, Y_model1)
#
#
#print(linear_model1.mse_(Yscore, Y_model1))
#print(mean_squared_error(Yscore, Y_model1))
#
#linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
#Y_model2 = linear_model2.predict_(Xpill)
#print(linear_model2.mse_(Yscore, Y_model2))
#print(mean_squared_error(Yscore, Y_model2))


data = pd.read_csv("../../module02/spacecraft_data.csv")
X = np.array(data[['Age']])
Y = np.array(data[['Sell_price']])
myLR_age = MyLinearRegression(thetas=[[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
myLR_age.fit_(X[:,0].reshape(-1,1), Y)
print(myLR_age.mse_(X[:,0].reshape(-1,1),Y))
