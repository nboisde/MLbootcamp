import numpy as np

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
		for i in range(self.max_iter):
			self.theta = self.theta - self.alpha * self.gradient(x, y)
		return self.theta

	@staticmethod
	def loss_(y, y_hat):
		if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
			raise TypeError("y and y_hat should be numpy arrays")
		if y.ndim != y_hat.ndim:
			raise ValueError("y and y_hat must have same dim")
		if y.shape[0] != y.shape[1]:
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


X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
print(mylr.predict_(X))
print(mylr.loss_elem_(mylr.predict_(X),Y))
print(mylr.loss_(mylr.predict_(X),Y))

mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.theta)
print(mylr.predict_(X))