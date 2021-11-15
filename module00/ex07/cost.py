import numpy as np

def add_intercept(x):
	if type(x) != type(np.ndarray([])):
		return None
	if x.ndim == 1:
		x = x.reshape(x.size, 1)
	x = np.insert(x, 0, 1, axis=1)
	return x

def predict(x, theta):
	if type(x) != type(np.ndarray([])):
		return None
	x = add_intercept(x)
	y_hat = x.dot(theta)
	return y_hat

def cost_elem(y, y_hat):
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
		tmp = (1/(2*M)) * (y_hat_val - y_val)**2
		under.append(tmp)
		tab.append(under)
	res = np.array(tab)
	return res

def cost_(y, y_hat):
	if type(y) != type(np.ndarray([])) or type(y_hat) != type(np.ndarray([])):
		return None
	if y.size != y_hat.size:
		return None
	mat = cost_elem(y, y_hat)
	J_val = 0
	for val in mat:
		J_val += val
	return J_val[0]

x1 = np.array([[0.], [1.], [2.], [3.], [4.]]) 
theta1 = np.array([[2.], [4.]])
y_hat1 = predict(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
# Example 1:
print(cost_elem(y1, y_hat1))
print(cost_(y1, y_hat1))

x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])
print(cost_(y2, y_hat2))

x3 = np.array([0, 15, -9, 7, 12, 3, -21]) 
theta3 = np.array([[0.], [1.]])
y_hat3 = predict(x3, theta3)
y3 = np.array([2, 14, -13, 5, 12, 4, -19])
print(cost_(y3, y_hat3))
print(cost_(y3, y3))
