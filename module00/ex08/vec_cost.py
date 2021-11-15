import numpy as np

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
	

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(cost_(X, Y))
print(cost_(X, X))