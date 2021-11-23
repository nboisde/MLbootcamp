import numpy as np

def add_polynomial_features(x, power):
	if not isinstance(x, np.ndarray) or x == np.array([]):
		return None
	if not isinstance(power, int) or power <= 0:
		return None
	c = x
	i = 2
	while i <= power:
		c = np.c_[c, x**i]
		i += 1
	return c

x = np.arange(1,11).reshape(5, 2)
print(add_polynomial_features(x, 3))
print(add_polynomial_features(x, 4))
print(add_polynomial_features(x, 5))
