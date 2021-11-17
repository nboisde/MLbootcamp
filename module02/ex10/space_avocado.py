import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
from data_splitter import data_splitter
from polynomial_model import add_polynomial_features

d = pd.read_csv("../space_avocado.csv")

#univariate model funciton
def univariate_model(data, feature1, target, proportion=0.80, alph=1e-7, max_it=100000):
	X = np.array(data[[feature1]])
	#print(X)
	Y = np.array(data[[target]])
	clean_set = data_splitter(X, Y, proportion)
	Y_TRAIN = clean_set[2]
	X_TRAIN = clean_set[0]
	Y_TEST = clean_set[3]
	X_TEST = clean_set[1]
	myLR_f = MyLR(theta = [[0.0], [0.0]], alpha=alph, max_iter=max_it)
	myLR_f.theta = myLR_f.normal_equation(X_TRAIN, Y_TRAIN)
	myLR_f.fit_(X_TRAIN, Y_TRAIN)
	print(myLR_f.theta)
	print(myLR_f.mse_(myLR_f.predict_(X_TEST), Y_TEST))

def univariate_polynomial_model(data, feature1, target, proportion=0.80, alph=1e-8, max_it=100000, polyfeat=2):
	X = np.array(data[[feature1]])
	print(X)
	X = add_polynomial_features(X, polyfeat)
	print(X)
	#print(X)
	Y = np.array(data[[target]])
	clean_set = data_splitter(X, Y, proportion)
	Y_TRAIN = clean_set[2]
	X_TRAIN = clean_set[0]
	Y_TEST = clean_set[3]
	X_TEST = clean_set[1]
	myLR_f = MyLR(theta = [[0.0], [0.0], [0.0]], alpha=alph, max_iter=max_it)
	#myLR_f.theta = myLR_f.normal_equation(X_TRAIN, Y_TRAIN)
	myLR_f.fit_(X_TRAIN, Y_TRAIN)
	print(myLR_f.theta)
	print(myLR_f.mse_(myLR_f.predict_(X_TEST), Y_TEST))

#univariate_model(d, 'weight', 'target')
#univariate_model(d, 'prod_distance', 'target')
#univariate_model(d, 'time_delivery', 'target')
univariate_polynomial_model(d, 'weight', 'target', polyfeat=2, alph=1e-7)