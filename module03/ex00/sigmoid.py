import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid_(x):
	if not isinstance(x, np.ndarray):
		raise TypeError("x must be a np array")
	sgm = lambda t: 1 / (1 + math.exp(-t))
	if x.ndim == 0:
		sigmoid = np.array(sgm(x))
		ret = np.append([], sigmoid)
	elif x.ndim == 1:
		sigmoid = np.array([sgm(xi) for xi in x])
		ret = sigmoid.reshape(-1, 1)
	else:
		sigmoid = []
		for val in x:
			for elem in val:
				tmp = np.array(sgm(elem))
				sigmoid.append(tmp)
		sigmoid = np.array(sigmoid)
		ret = sigmoid.reshape(-1, 1)
	return ret

x = np.array(2)
print(sigmoid_(x))

x = np.array(-4)
print(sigmoid_(x))

x = np.array([[-4], [2], [0]])
print(sigmoid_(x))

CONT_X = np.arange(-10, 10, 0.01).reshape(-1, 1)
CONT_Y = sigmoid_(CONT_X)
plt.plot(CONT_X, CONT_Y, color='#F0D705')
plt.show()
