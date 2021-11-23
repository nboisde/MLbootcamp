import numpy as np

def data_splitter_one_for_all(x, y, xfeat=3, yclass=4, proportion=0.80):
    if not isinstance(x, np.ndarray):
        return None
    if not isinstance(y, np.ndarray):
        return None
    if x.ndim != 2 or (y.ndim != 1 and y.ndim != 2):
        return None
    if y.ndim == 1:
        if y.shape[0] != x.shape[1]:
            print(4)
            return None
    if y.ndim == 2:
        if y.shape[0] != x.shape[0]:
            return None
    if not isinstance(proportion, float):
        return None
    if proportion > 1 or proportion < 0:
        return None
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    tmp = np.c_[x, y]
    rng = np.random.default_rng()
    rng.shuffle(tmp)
    split_horizontally_idx = int(tmp.shape[0] * proportion)
    train = tmp[:split_horizontally_idx , :]
    test = tmp[split_horizontally_idx: , :]
    x_train = train
    x_test = test
    for i in range(yclass):
        x_train = np.delete(x_train, -1, axis=1)
        x_test = np.delete(x_test, -1, axis=1)
    y_train = train
    y_test = test
    for i in range(xfeat):
        y_train = np.delete(y_train, 0, axis=1)
        y_test = np.delete(y_test, 0, axis=1)
    #print(y_train)
    return (x_train, x_test, y_train, y_test)

# x feat includes and remove the index...
def data_splitter_one_for_all_v2(d, xfeat=3, yclass=4, proportion=0.80):
    if not isinstance(d, np.ndarray):
        return None
    if not isinstance(proportion, float):
        return None
    if proportion > 1 or proportion < 0:
        return None
    rng = np.random.default_rng()
    rng.shuffle(d)
    split_horizontally_idx = int(d.shape[0] * proportion)
    train = d[:split_horizontally_idx , :]
    test = d[split_horizontally_idx: , :]
    id_train = train
    id_test = test
    for i in range(yclass + xfeat + 1):
        id_train = np.delete(id_train, -1, axis=1)
        id_test = np.delete(id_test, -1, axis=1)
    x_train = train
    x_test = test
    for i in range(yclass + 1):
        x_train = np.delete(x_train, -1, axis=1)
        x_test = np.delete(x_test, -1, axis=1)
    x_train = np.delete(x_train, 0, axis=1)
    x_test = np.delete(x_test, 0, axis=1)
    y_train = train
    y_test = test
    for i in range(xfeat):
        y_train = np.delete(y_train, 0, axis=1)
        y_test = np.delete(y_test, 0, axis=1)
    y_train = np.delete(y_train, -1, axis=1)
    y_test = np.delete(y_test, -1, axis=1)
    real_train = train[:, -1].reshape(-1, 1)
    real_test = test[:, -1].reshape(-1, 1)
    return (id_train, id_test, x_train, x_test, y_train, y_test, real_train, real_test)

def data_splitter_one_for_all_dslr(d, xfeat=3, yclass=4, proportion=0.80):
	if not isinstance(d, np.ndarray):
		return None
	if not isinstance(proportion, float):
		return None
	if proportion > 1 or proportion < 0:
		return None
	rng = np.random.default_rng()
	rng.shuffle(d)
	split_horizontally_idx = int(d.shape[0] * proportion)
	train = d[:split_horizontally_idx , :]
	test = d[split_horizontally_idx: , :]
	id_train = train
	id_test = test
	for i in range(yclass + xfeat):
		id_train = np.delete(id_train, -1, axis=1)
		id_test = np.delete(id_test, -1, axis=1)
	x_train = train
	x_test = test
	for i in range(yclass):
		x_train = np.delete(x_train, -1, axis=1)
		x_test = np.delete(x_test, -1, axis=1)
	x_train = np.delete(x_train, 0, axis=1)
	x_test = np.delete(x_test, 0, axis=1)
	y_train = train
	y_test = test
	for i in range(xfeat + 1):
		y_train = np.delete(y_train, 0, axis=1)
		y_test = np.delete(y_test, 0, axis=1)
	return (id_train, id_test, x_train, x_test, y_train, y_test)