import numpy as np
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

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

def mse_(y, y_hat):
	return (2 * cost_(y, y_hat))

def rmse_(y, y_hat):
	return (math.sqrt(mse_(y, y_hat)))

def mae_(y, y_hat):
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
		tmp = (1/(M)) * abs(y_hat_val - y_val)
		under.append(tmp)
		tab.append(under)
	mat = np.array(tab)
	J_val = 0
	for val in mat:
		J_val += val
	return J_val[0]

def r2score_(y, y_hat):
	if type(y) != type(np.ndarray([])) or type(y_hat) != type(np.ndarray([])):
		return None
	if y.size != y_hat.size:
		return None
	ym = 0
	for v in y:
		ym += (v - np.mean(y))**2
	M = y.size
	return 1 - ((M * mse_(y, y_hat)) / ym)

x = np.array([0, 15, -9, 7, 12, 3, -21])
y = np.array([2, 14, -13, 5, 12, 4, -19])
print("mean square error : " + str(mse_(x,y)))
print("sqrt mean square error : " + str(rmse_(x,y)))
print("mean absolute error : " + str(mae_(x, y)))
print("r2 score : " + str(r2score_(x,y)))
