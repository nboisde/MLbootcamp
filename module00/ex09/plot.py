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

x = np.arange(1,6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1= np.array([18,-1])
plot_with_cost(x, y, theta1)

theta2 = np.array([14, 0])
plot_with_cost(x, y, theta2)

#theta3 = np.array([12, 0.8])
#plot_with_cost(x, y, theta3)