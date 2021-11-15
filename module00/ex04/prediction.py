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

x = np.arange(1,6)
theta1 = np.array([5, 0])

print(predict_(x, theta1))
theta2 = np.array([0, 1])
print(predict_(x, theta2))

theta3 = np.array([5, 3])
print(predict_(x, theta3))

theta4 = np.array([-3, 1])
print(predict_(x, theta4))