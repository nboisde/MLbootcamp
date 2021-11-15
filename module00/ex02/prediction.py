import numpy as np

def simple_predict(x, theta):
	#y_hat = np.ndarray([])
	if type(theta) != type(np.ndarray([])) or theta.size != 2:
		return None
	if type(x) != type(np.ndarray([])) or x.size == 0:
		return None
	y_hat = []
	for val in x:
		p = theta[0] + theta[1] * val
		y_hat.append(float(p))
	ret = np.array(y_hat)
	return ret

x = np.arange(1,6)
theta1 = np.array([5, 0])

print(simple_predict(x, theta1))
theta2 = np.array([0, 1])
print(simple_predict(x, theta2))

theta3 = np.array([5, 3])
print(simple_predict(x, theta3))

theta4 = np.array([-3, 1])
print(simple_predict(x, theta4))

def simple_predict_opti(x, theta):
	if type(theta) != type(np.ndarray([])) or theta.size != 2:
		return None
	if type(x) != type(np.ndarray([])) or x.size == 0:
		return None
	x = x.reshape(x.size, 1)
	x_bis = np.full((x.size, 2), 1)
	print (x)
	print(x_bis)
	for y in range(x_bis.shape[0]):
		x_bis[y][1] *= float(x[y][0])

	theta = theta.reshape(theta.size, 1)
	y_hat = x_bis.dot(theta)
	y_hat = y_hat.transpose()
	res = y_hat[0]
	return res

print(simple_predict_opti(x, theta3))
