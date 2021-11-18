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
    print(y_train)
    return (x_train, x_test, y_train, y_test)
