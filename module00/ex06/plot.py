import matplotlib.pyplot as plt 
import numpy as np

def add_intercept(x):
	if type(x) != type(np.ndarray([])):
		return None
	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	x = np.insert(x, 0, 1, axis=1)
	return x

def predict_(x, theta):
	if type(x) != type(np.ndarray([])):
		return None
	x = add_intercept(x)
	y_hat = x.dot(theta)
	return y_hat

def plot(x, y, theta):
	#ybis = theta[0] + theta[1] * xbis
	y_hat = predict_(x, theta)
	print(y_hat)
	plt.plot(x, y_hat, 'r')
	plt.scatter(x, y)
	plt.show()

x = np.arange(1,6)
y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
#theta1 = np.array([4.5, -0.2])
#plot(x, y, theta1)

#theta2 = np.array([-1.5, 2])
#plot(x, y, theta2)

theta3 = np.array([3, 0.3])
plot(x, y, theta3)