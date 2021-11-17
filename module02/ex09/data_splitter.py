import numpy as np

def data_splitter(x, y, proportion):
    if not isinstance(x, np.ndarray):
        return None
    if not isinstance(y, np.ndarray):
        return None
    if x.ndim != 2 or y.ndim != 1:
        return None
    if y.shape[0] != x.shape[1]:
        return None
    if not isinstance(proportion, float):
        return None
    if proportion > 1 or proportion < 0:
        return None
    y = y.reshape(-1, 1)
    tmp = np.c_[x, y]
    rng = np.random.default_rng()
    rng.shuffle(tmp)
    split_horizontally_idx = int(tmp.shape[0] * proportion)
    train = tmp[:split_horizontally_idx , :]
    test = tmp[split_horizontally_idx: , :]
    x_train = train[:,:x.shape[0]]
    x_test = test[:,:x.shape[0]]
    y_train = train[:, :-1]
    y_test = test[:, :-1]
    return (x_train, x_test, y_train, y_test)
    
    

x = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [4, 5, 6, 7], [10, 10, 10, 10]])
y = np.array([20, 21, 22, 23])
print(data_splitter(x, y, 0.80))