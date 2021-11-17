import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyLinearRegression():
	def __init__(self, theta, alpha=0.001, max_iter=10000):
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

	def gradient(self, x, y):
		if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(self.theta, np.ndarray):
			raise TypeError("varaibles must be np.arrays")
		if y.shape[0] != x.shape[0] or x.shape[1] + 1 != self.theta.shape[0]:
			raise ValueError("impossible to calculate due to error dimentions")
		x = np.c_[np.ones(x.shape[0]), x]
		return ((1/y.size) * np.transpose(x).dot(x.dot(self.theta) - y))

	def predict_(self, x):
		if not isinstance(x, np.ndarray) or not isinstance(self.theta, np.ndarray):
			raise TypeError("should be numpy arrays")
		if x.ndim != 2:
			raise ValueError("x must be a matrix")
		theta = self.theta.reshape((len(self.theta), 1))
		x = np.c_[np.ones(x.shape[0]), x]
		if theta.shape[0] != x.shape[1]:
			raise ValueError("multiplictation impossible...")
		return (x.dot(theta))

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
		#print(self.theta)
		for i in range(self.max_iter):
			self.theta = self.theta - self.alpha * self.gradient(x, y)
		return self.theta

	@staticmethod
	def loss_(y, y_hat):
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
			raise TypeError("y and y_hat should be numpy arrays")
		if y.ndim != y_hat.ndim:
			raise ValueError("y and y_hat must have same dim")
		if y.shape[0] != y.shape[1]:# and y.shape[0] == y_hat.shape[0]:
			tmp = np.transpose(y - y_hat)
		else:
			tmp = y - y_hat
		return (1/(2 * y.size)) * (tmp).dot((y - y_hat))[0][0]

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
		return np.array(res)

	def mse_(self, y, y_hat):
		return (2 * self.loss_(y, y_hat))
	
	def plot_uni(self, x, y, y_hat, xax="x axis", yax="y axis", yleg="data", yhleg="predicted", coly='blue', colyh='lightblue'):
		plt.xlabel(xax)
		plt.ylabel(yax)
		plt.scatter(x, y, label=yleg, color=coly)
		plt.scatter(x, y_hat, label=yhleg, color=colyh)
		plt.legend()
		plt.grid()
		plt.show()
	

# UNIVARIATE MODEL WITH AGE.
#print("----------UNIVARIATE LINEAR REGRESSION ON AGE-----------")
#data = pd.read_csv("../spacecraft_data.csv")
#X = np.array(data[['Age']])
#Y = np.array(data[['Sell_price']])
#myLR_age = MyLinearRegression(theta = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
#myLR_age.fit_(X[:,0].reshape(-1,1), Y)
#print(myLR_age.mse_(myLR_age.predict_(X[:,0].reshape(-1,1)),Y))
#myLR_age.plot_uni(X, Y, myLR_age.predict_(X), "x1: age in years", "y: sell price (in keuros)", "sell price", "predicted sell price")

# UNIVARIATE MODEL WITH THRUST.
#print("----------UNIVARIATE LINEAR REGRESSION ON THRUST-----------")
#data = pd.read_csv("../spacecraft_data.csv")
#X2 = np.array(data[['Thrust_power']])
#Y2 = np.array(data[['Sell_price']])
#myLR_th = MyLinearRegression(theta = [[500.0], [3.0]], alpha = 2e-5, max_iter = 400000)
#myLR_th.fit_(X2[:,0].reshape(-1,1), Y2)
#print(myLR_th.mse_(myLR_th.predict_(X2[:,0].reshape(-1,1)),Y2))
#myLR_th.plot_uni(X2, Y2, myLR_th.predict_(X2), "x1: age in years", "y: sell price (in keuros)", "sell price", "predicted sell price")

# UNIVARIATE MODEL WITH THRUST.
#print("----------UNIVARIATE LINEAR REGRESSION ON DISTANCE-----------")
#data = pd.read_csv("../spacecraft_data.csv")
#X3 = np.array(data[['Terameters']])
#Y3 = np.array(data[['Sell_price']])
#myLR_dis = MyLinearRegression(theta = [[500.0], [2.0]], alpha = 2e-5, max_iter = 400000)
#myLR_dis.fit_(X3[:,0].reshape(-1,1), Y3)
#print(myLR_dis.mse_(myLR_dis.predict_(X3[:,0].reshape(-1,1)),Y3))
#myLR_dis.plot_uni(X3, Y3, myLR_dis.predict_(X3), "x1: age in years", "y: sell price (in keuros)", "sell price", "predicted sell price")


#print("----------MULTIVARIATE LINEAR REGRESSION-----------")
#data = pd.read_csv("../spacecraft_data.csv")
#X = np.array(data[['Age','Thrust_power','Terameters']])
#Y = np.array(data[['Sell_price']])
#my_lreg = MyLinearRegression(theta = [250.0, -20.0, 4.0, -2.0], alpha = 2.3e-5, max_iter = 620000)
#print(my_lreg.mse_(X,Y))
#
#my_lreg.fit_(X,Y)
#print(my_lreg.theta)
#print(my_lreg.mse_(my_lreg.predict_(X),Y))
#Xage = np.array(data[['Age']])
#my_lreg.plot_uni(Xage, Y, my_lreg.predict_(X), "x1: age in years", "y: sell price (in keuros)", "sell price", "predicted sell price")
#Xth = np.array(data[['Thrust_power']])
#my_lreg.plot_uni(Xth, Y, my_lreg.predict_(X), "x1: Thrust power", "y: sell price (in keuros)", "sell price", "predicted sell price", coly='green', colyh='lightgreen')
#Xmet = np.array(data[['Terameters']])
#my_lreg.plot_uni(Xmet, Y, my_lreg.predict_(X), "x1: Thrust power", "y: sell price (in keuros)", "sell price", "predicted sell price", coly='purple', colyh='#b19cd9')

# plot picturized are calculated with final theta values of 
#theta = [[333.93924232], [-22.49725381], [5.86129269], [-2.58474427]]
