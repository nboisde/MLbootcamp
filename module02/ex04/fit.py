import numpy as np

def gradient(x, y, theta):
	if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
		raise TypeError("varaibles must be np.arrays")
	if y.shape[0] != x.shape[0] or x.shape[1] + 1 != theta.shape[0]:
		raise ValueError("impossible to calculate due to error dimentions")
	x = np.c_[np.ones(x.shape[0]), x]
	return ((1/y.size) * np.transpose(x).dot(x.dot(theta) - y))

def predict_(x, theta):
	if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
		raise TypeError("should be numpy arrays")
	if x.ndim != 2:
		raise ValueError("x must be a matrix")
	theta = theta.reshape((len(theta), 1))
	x = np.c_[np.ones(x.shape[0]), x]
	if theta.shape[0] != x.shape[1]:
		raise ValueError("multiplictation impossible...")
	return (x.dot(theta))

def fit_(x, y, theta, alpha=0.001, max_iter=10000):
	if not isinstance(alpha, (float, int)):
		raise TypeError("alpha and max_iter should be numbers")
	if alpha < 0 or alpha > 1:
		raise ValueError("alpha should be between 0 and 1")
	if not isinstance(max_iter, int):
		raise TypeError("max_iter should be an int")
	if max_iter < 0:
		raise ValueError("iteration number must be positive")
	for i in range(max_iter):
		theta = theta - alpha * gradient(x, y, theta)
	return theta

x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])

theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
print(theta2)

print(predict_(x, theta2))