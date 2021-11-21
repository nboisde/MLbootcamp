from numpy.lib.index_tricks import diag_indices_from
import pandas as pd
import numpy as np

# This function is clearly enhancable. We must carry about labels values, Should be present in unique
# also, about columns order in the final display tab...
def confusion_matrix_(y, y_hat, labels=None, df_option=False):
	if not (labels == None or isinstance(labels, list)) or not isinstance(df_option, bool):
		return None
	unique, counts = np.unique(np.c_[y, y_hat], return_counts=True)
	conf_mat = np.zeros((len(unique), len(unique)))
	for valy, valyh in zip(y, y_hat):
		y_i, yh_i = 0, 0
		for i, u in enumerate(unique):
			if valy == u:
				y_i = i
			if valyh == u:
				yh_i = i
		conf_mat[y_i][yh_i] += 1
	if labels != None:
		if len(labels) < 2 or len(labels) > len(unique):
			return None
		for elem in labels:
			if labels.count(elem) > 1:
				return None
		for val in labels:
			if not val in unique:
				return None
		tmp = []
		for u in unique:
			if u in labels:
				tmp.append(u)
		labels = tmp
		for i, u in enumerate(unique):
			if u not in labels:
				conf_mat = np.delete(conf_mat, i, axis=1)
				conf_mat = np.delete(conf_mat, i, axis=0)
		col_name = labels
	else:
		col_name = unique
	if df_option == True:
		df = pd.DataFrame(conf_mat, index=col_name, columns=col_name)
		return df
	else:
		return conf_mat

					
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet'], df_option=True))
