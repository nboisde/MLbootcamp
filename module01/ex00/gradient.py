import numpy as np

def simple_gradient(x, y, theta):
	j1 = 0
	i = 0
	j2 = 0
	if theta.size != 2 or x.size != y.size:
		return None
	for xval, yval in np.nditer([x, y]):
		i += 1
		j2 += ((theta[0] + (theta[1] * xval)) - yval) * xval
		j1 += ((theta[0] + (theta[1] * xval)) - yval)
	j1 = j1 / i
	j2 = j2 / i
	tmp = [j1, j2]
	return np.array(tmp)

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
# Example 0:
theta1 = np.array([2, 0.7])
print(simple_gradient(x, y, theta1))

theta2 = np.array([1, -0.4])
print(simple_gradient(x, y, theta2))