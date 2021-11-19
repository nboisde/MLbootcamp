from pandas.core.generic import NDFrame
import numpy as np

def transform_pred_to_final_values(y_hat):
	"""Transform the prediction in zeros and one to comparate"""
	yh2 = y_hat
	for yh in yh2:
		yh = 1.0 if yh >= 0.5 else 0.0
	print(yh2)
	return yh2

# FALSE POSITIVES
def fp(y, y_hat, pos_label=1):
	fp = 0
	for vy, vyh in zip(y, y_hat):
		if vyh != vy and vy != pos_label:
			fp += 1
	return fp

#FALSE NEGATIVES
def fn(y, y_hat, pos_label=1):
	fn = 0
	for vy, vyh in zip(y, y_hat):
		if vyh != vy and vy == pos_label:
			fn += 1
	return fn

#TRUE POSITIVES
def tp(y, y_hat, pos_label=1):
	tp = 0
	for vy, vyh in zip(y, y_hat):
		if vyh == vy and vy == pos_label:
			tp += 1
	return tp

#TRUE NEGATIVES
def tn(y, y_hat, pos_label=1):
	tn = 0
	for vy, vyh in zip(y, y_hat):
		if vyh == vy and vy != pos_label:
			tn += 1
	return tn

def accuracy_score_(y, y_hat):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		raise TypeError("Should be np arrays")
	if y.shape != y_hat.shape:
		raise ValueError("Must be the same shape to score.")
	y_hat = transform_pred_to_final_values(y_hat)

y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
# Accuracy
## your implementation
print(accuracy_score_(y, y_hat))