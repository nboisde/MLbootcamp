from pandas.core.generic import NDFrame
import numpy as np

def transform_pred_to_final_values(y_hat):
	"""Transform the prediction in zeros and one to comparate"""
	yh2 = y_hat
	#print(y_hat)
	for yh in yh2:
		yh = 1.0 if yh >= 0.5 else 0.0
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

def accuracy_score_(y, y_hat, pos_label=1):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		raise TypeError("Should be np arrays")
	if y.shape != y_hat.shape:
		raise ValueError("Must be the same shape to score.")
	if (pos_label) == 1:
		y_hat = transform_pred_to_final_values(y_hat)
	tpa = tp(y, y_hat, pos_label)
	tna = tn(y, y_hat, pos_label)
	fpa = fp(y, y_hat, pos_label)
	fna = fn(y, y_hat, pos_label)
	return (tpa + tna) / (tpa + fpa + tna + fna)

def precision_score_(y, y_hat, pos_label=1):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		raise TypeError("Should be np arrays")
	if y.shape != y_hat.shape:
		raise ValueError("Must be the same shape to score.")
	if (pos_label) == 1:
		y_hat = transform_pred_to_final_values(y_hat)
	tpa = tp(y, y_hat, pos_label)
	tna = tn(y, y_hat, pos_label)
	fpa = fp(y, y_hat, pos_label)
	fna = fn(y, y_hat, pos_label)
	return tpa / (tpa + fpa)

def recall_score_(y, y_hat, pos_label=1):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		raise TypeError("Should be np arrays")
	if y.shape != y_hat.shape:
		raise ValueError("Must be the same shape to score.")
	if (pos_label) == 1:
		y_hat = transform_pred_to_final_values(y_hat)
	tpa = tp(y, y_hat, pos_label)
	tna = tn(y, y_hat, pos_label)
	fpa = fp(y, y_hat, pos_label)
	fna = fn(y, y_hat, pos_label)
	return tpa / (tpa + fna)

def f1_score_(y, y_hat, pos_label=1):
	if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
		raise TypeError("Should be np arrays")
	if y.shape != y_hat.shape:
		raise ValueError("Must be the same shape to score.")
	if (pos_label) == 1:
		y_hat = transform_pred_to_final_values(y_hat)
	return (2 * precision_score_(y, y_hat, pos_label) * recall_score_(y, y_hat, pos_label)) / (precision_score_(y, y_hat, pos_label) + recall_score_(y, y_hat, pos_label))

y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
# Accuracy
## your implementation
print(accuracy_score_(y, y_hat))
print(precision_score_(y, y_hat))
print(recall_score_(y, y_hat))
print(f1_score_(y, y_hat))

y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
print(accuracy_score_(y, y_hat, pos_label='dog'))
print(precision_score_(y, y_hat, pos_label='dog'))
print(recall_score_(y, y_hat, pos_label='dog'))
print(f1_score_(y, y_hat, pos_label='dog'))

print(accuracy_score_(y, y_hat, pos_label='norminet'))
print(precision_score_(y, y_hat, pos_label='norminet'))
print(recall_score_(y, y_hat, pos_label='norminet'))
print(f1_score_(y, y_hat, pos_label='norminet'))