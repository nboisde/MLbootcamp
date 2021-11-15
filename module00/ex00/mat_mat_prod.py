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

def mat_mat_prod(A, B):
	if A.ndim == 1 or B.ndim == 1:
		return None
	if A.shape[1] != B.shape[0]:
		return None

	res = []
	for x in range(A.shape[0]):
		tmp = []
		for y in range(B.shape[1]):
			val = 0
			for cursor in range(A.shape[1]):
				val += A[x][cursor] * B[cursor][y]
			tmp.append(val)
		res.append(tmp)
	
	arr = np.array(res)
	return arr

			
		
		


W = np.array([
[ -8, 8, -6, 14, 14, -9, -4],
[ 2, -11, -2, -11, 14, -2, 14], [-13, -2, -5, 3, -8, -4, 13],
[ 2, 13, -14, -15, -14, -15, 13], [ 2, -1, 12, 3, -7, -3, -6]])

Z = np.array([
[ -6, -1, -8, 7, -8],
[ 7, 4, 0, -10, -10], [ 7, -13, 2, 2, -11], [ 3, 14, 7, 7, -4],
[ -1, -3, -8, -4, -14], [ 9, -14, 9, 12, -7], [ -9, -4, -10, -3, 6]])

print(mat_mat_prod(W, Z))
print(W.dot(Z))
print(mat_mat_prod(Z, W))
print(Z.dot(W))