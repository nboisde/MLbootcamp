import numpy as np

def add_intercept(x):
	if type(x) != type(np.ndarray([])):
		return None
	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	x = np.insert(x, 0, 1, axis=1)
	return x

def gradient(x, y, theta):
	nabla = 0
	xi = add_intercept(x)
	xt = np.transpose(xi)
	M = xt.shape[1]
	nabla = (xt.dot(np.matmul(xi, theta) - y))/M
	return nabla

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
theta1 = np.array([2, 0.7])
print(gradient(x, y, theta1))

theta2 = np.array([1, -0.4])
print(gradient(x, y, theta2))