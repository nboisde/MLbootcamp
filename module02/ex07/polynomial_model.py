import numpy as np

def add_polynomial_features(x, power):
	if not isinstance(power, int):
		raise TypeError("power must be an int")
	if not isinstance(x, np.ndarray):
		raise TypeError("x, must be a np array")
	if power < 1:
		raise ValueError("power must be positive")
	return np.fliplr(np.column_stack([x**((power + 1) - 1 - i) for i in range(power)]))

x = np.arange(1,6).reshape(-1, 1)
print(add_polynomial_features(x, 3))
print(add_polynomial_features(x, 6))