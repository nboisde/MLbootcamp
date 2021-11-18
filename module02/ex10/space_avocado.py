import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zscore import zscore
from minmax import minmax
from mylinearregression import MyLinearRegression as MyLR
from data_splitter import data_splitter
from polynomial_model import add_polynomial_features


d = pd.read_csv("../space_avocado.csv")
d.head()
X = np.array(d[['weight', 'prod_distance', 'time_delivery']])
Y = np.array(d[['target']])
Xm = minmax(X)
Ym = minmax(Y)
x0 = add_polynomial_features(Xm[:,0], 2)
x1 = add_polynomial_features(Xm[:,1], 3)
x2 = add_polynomial_features(Xm[:,2], 1)
Xl = np.c_[x0, x1, x2]
print(Xl)
clean_set = data_splitter(Xl, Ym, 0.80)
Y_TRAIN = clean_set[2]
X_TRAIN = clean_set[0]
Y_TEST = clean_set[3]
X_TEST = clean_set[1]
#plt.scatter(X_TRAIN[:,1], Y_TRAIN)
#plt.show()
#plt.scatter(X_TRAIN[:,0], Y_TRAIN)
#plt.show()
#plt.scatter(X_TRAIN[:,2], Y_TRAIN)
#plt.show()
myLR_f = MyLR([[0],[0],[0],[0],[0],[0],[0]])
myLR_f.theta = myLR_f.normal_equation(X_TRAIN, Y_TRAIN)
print(f"theta : {myLR_f.theta}")
print(myLR_f.mse_(myLR_f.predict_(X_TEST), Y_TEST))
#myLR_f.plot_uni(X_TEST[:,0], Y_TEST, myLR_f.predict_(X_TEST), "x1: weight", "y: target(minmax normalization)", "target", "predicted target")
#myLR_f.plot_uni(X_TEST[:,2], Y_TEST, myLR_f.predict_(X_TEST), "x1: prod_distance", "y: target(minmax normalization)", "target", "predicted target", coly='green', colyh='lightgreen')
#myLR_f.plot_uni(X_TEST[:,5], Y_TEST, myLR_f.predict_(X_TEST), "x1: time_delivery", "y: target(minmax normalization)", "target", "predicted target", coly='purple', colyh='#b19cd9')