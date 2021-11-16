import numpy as np

def simple_predict_v1(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		raise TypeError("should be numpy arrays")
	if x.ndim != 2:
		raise ValueError("x must be a matrix")
	if theta.ndim != 1:
		raise ValueError("x must be a matrix")
	if len(theta) != x.shape[0]:
		raise ValueError("multiplictation impossible...")
	#theta = theta.reshape((len(theta), 1))
	print(theta)
	#x = np.transpose(x)
	x = np.c_[np.ones(x.shape[0]), x]
	return (x.dot(theta))

def simple_predict(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		raise TypeError("should be numpy arrays")
	if x.ndim != 2:
		raise ValueError("x must be a matrix")
	theta = theta.reshape((len(theta), 1))
	x = np.c_[np.ones(x.shape[0]), x]
	if theta1.shape[0] != x.shape[1]:
		raise ValueError("multiplictation impossible...")
	return (x.dot(theta))

x = np.arange(1,13).reshape((4, -1))
#print(x)
theta1 = np.array([5, 0, 0, 0])
print(simple_predict(x, theta1))

x = np.arange(1, 13).reshape(-1, 2)
theta1 = np.ones(3).reshape(-1, 1)
print(simple_predict(x, theta1))
