import numpy as np
import matplotlib.pyplot as plt 

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

def oth_grd(x, y, theta):
	arrx = np.append([], x)
	arry = np.append([], y)
	#print(arrx)
	#print(arry)
	return (gradient(arrx, arry, theta))

def predict_(x, theta):
	if type(x) != type(np.ndarray([])):
		return None
	x = add_intercept(x)
	y_hat = x.dot(theta)
	return y_hat

def cost_(y, y_hat):
	if type(y) != type(np.ndarray([])) or type(y_hat) != type(np.ndarray([])):
		return None
	if y.size != y_hat.size:
		return None
	M = y.size
	tab = []
	for yv, yhv in np.nditer([y, y_hat]):
		tab.append(yhv - yv)
	vec = np.array(tab)
	return (1/(2*M))*(vec.dot(vec))

def plot_with_cost(x, y, theta):
	#ybis = theta[0] + theta[1] * xbis
	ones = np.array([])
	for i in y:
		ones = np.insert(ones, 0, 0, axis=0)
	print(ones)
	y_hat = predict_(x, theta)
	print(y_hat)
	plt.plot(x, y_hat, 'r')
	plt.scatter(x, y)
	plt.vlines(x, y, y_hat, 'g', '--')
	plt.title("Cost: " + str(format(2 * cost_(y, y_hat), '.6f')))
	plt.show()

def fit_(x, y, theta, alpha, max_iter):
	print(theta)
	t1 = theta[0] - alpha * oth_grd(x, y, theta)[0]
	t2 = theta[1] - alpha * oth_grd(x, y, theta)[1]
	lol = np.append([], [t1, t2])
	for i in range(max_iter - 1):
		#print(i)
		t1 = t1 - alpha * oth_grd(x, y, lol)[0]	
		t2 = t2 - alpha * oth_grd(x, y, lol)[1]
		lol = np.append([], [t1, t2])
		#print(theta)
	return lol	


x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta= np.array([1, 1])
theta1 = fit_(x, y, theta, alpha=5e-4, max_iter=300000)
print(theta1)

arrx = np.append([], x)
arry = np.append([], y)
plot_with_cost(arrx, arry, theta1)
print(predict_(x, theta1))

