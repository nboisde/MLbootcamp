import numpy as np

def vec_gradient(x, y, theta):
	xt = np.transpose(x)
	thetax = x.dot(theta)
	thetax = np.transpose(thetax)
	thetax = thetax - y
	m = y.size
	thetax = xt.dot(thetax)
	return (1/m) * thetax	

X = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1], [ -8, -4, 6], [ -5, -9, 6], [ 1, -5, 11], [ 9,-11, 8]])

Y = np.array([2, 14, -13, 5, 12, 4, -19])
theta = np.array([3, 0.5, -6])
print(vec_gradient(X, Y, theta))

theta = np.array([0, 0, 0])
print(vec_gradient(X, Y, theta))

print(vec_gradient(X, X.dot(theta), theta))