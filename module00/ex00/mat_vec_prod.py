import numpy as np

def dot(x, y):
	if x.size == 0 or y.size == 0 or x.size != y.size:
		return None
	dot = 0
	for val, val2 in np.nditer([x, y]):
		dot = dot + (val*val2)
	return float(dot)

def mat_vec_prod(M, y):
	Ms = M.shape[1]
	ys1 = y.shape[0]
	ys2 = y.shape[1]
	if Ms == 0 or ys1 == 0 or Ms != ys1:
		return None
	y = y.reshape(ys2, ys1)[0]
	tmp = []
	for arr in M:
		tmp.append(dot(arr, y))
	print(tmp)
	res = np.array(tmp)
	res = res.reshape(res.size, 1)
	return res

W = np.array([
[ -8, 8, -6, 14, 14, -9, -4],
[ 2, -11, -2, -11, 14, -2, 14], [-13, -2, -5, 3, -8, -4, 13],
[ 2, 13, -14, -15, -14, -15, 13], [ 2, -1, 12, 3, -7, -3, -6]])

X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7,1))

print(mat_vec_prod(W, X))
print(W.dot(X))
