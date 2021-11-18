import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zscore import zscore
from minmax import minmax
from mylinearregression import MyLinearRegression as MyLR
from data_splitter import data_splitter
from polynomial_model import add_polynomial_features


#univariate model funciton
def univariate_model(data, feature1, target, proportion=0.80, alph=1e-7, max_it=100000, normal=0, model_name="univariate_polynomial_model", norm=0):
	X = np.array(data[[feature1]])
	Y = np.array(data[[target]])
	if norm == 0:
		Xm=X
		Xm=Y
	elif norm == 1:
		Xm = minmax(X)
		Ym = minmax(Y)
	else:
		Xm = zscore(X)
		Ym = zscore(Y)
	clean_set = data_splitter(Xm, Ym, proportion)
	Y_TRAIN = clean_set[2]
	X_TRAIN = clean_set[0]
	Y_TEST = clean_set[3]
	X_TEST = clean_set[1]
	myLR_f = MyLR(theta = [[0.0], [0.0]], alpha=alph, max_iter=max_it)
	if normal == 1:
		myLR_f.theta = myLR_f.normal_equation(X_TRAIN, Y_TRAIN)
	else:
		myLR_f.fit_(X_TRAIN, Y_TRAIN)
	print(f"{model_name} feat: {feature1}")
	print(f"theta : {myLR_f.theta}")
	print(myLR_f.mse_(myLR_f.predict_(X_TEST), Y_TEST))

#univariate poly model
def univariate_polynomial_model(data, feature1, t, proportion=0.80, alph=1e-8, max_it=100000, polyfeat=2, th=[[0.0], [0.0]], normal=0, model_name="univariate_polynomial_model", norm=0):
	X = np.array(data[[feature1]])
	X = add_polynomial_features(X, polyfeat)
	Y = np.array(data[[t]])
	if norm == 0:
		Xm=X
		Xm=Y
	elif norm == 1:
		Xm = minmax(X)
		Ym = minmax(Y)
	else:
		Xm = zscore(X)
		Ym = zscore(Y)
	clean_set = data_splitter(Xm, Ym, proportion)
	Y_TRAIN = clean_set[2]
	X_TRAIN = clean_set[0]
	Y_TEST = clean_set[3]
	X_TEST = clean_set[1]
	myLR_f = MyLR(th, alpha=alph, max_iter=max_it)
	if normal == 1:
		myLR_f.theta = myLR_f.normal_equation(X_TRAIN, Y_TRAIN)
	else:
		myLR_f.fit_(X_TRAIN, Y_TRAIN)
	print(f"{model_name} polynomial order: {polyfeat}, feat {feature1}")
	print(f"theta : {myLR_f.theta}")
	print(myLR_f.mse_(myLR_f.predict_(X_TEST), Y_TEST))

#multivariate model
def multivariate_model(data, features, t, proportion=0.80, alph=1e-8, max_it=100000, th=[[0.0], [0.0]], normal=0, model_name="multivariate_model", norm=0):
	if not isinstance(features, list):
		raise TypeError("Must have a feature")
	for val in features:
		if not isinstance(val, str):
			raise TypeError("features must be strings")
	X = np.array(data[features])
	Y = np.array(data[[t]])
	if norm == 0:
		Xm=X
		Xm=Y
	elif norm == 1:
		Xm = minmax(X)
		Ym = minmax(Y)
	else:
		Xm = zscore(X)
		Ym = zscore(Y)
	clean_set = data_splitter(Xm, Ym, proportion)
	Y_TRAIN = clean_set[2]
	X_TRAIN = clean_set[0]
	Y_TEST = clean_set[3]
	X_TEST = clean_set[1]
	myLR_f = MyLR(th, alpha=alph, max_iter=max_it)
	if normal == 1:
		myLR_f.theta = myLR_f.normal_equation(X_TRAIN, Y_TRAIN)
	else:
		myLR_f.fit_(X_TRAIN, Y_TRAIN)
	print(f"{model_name} features:{features}")
	print(f"theta : {myLR_f.theta}")
	print(myLR_f.mse_(myLR_f.predict_(X_TEST), Y_TEST))


if __name__ == "__main__":
	d = pd.read_csv("../space_avocado.csv")
	univariate_model(d, 'weight', 'target', normal=1, norm=1)
	univariate_model(d, 'prod_distance', 'target', normal=1, norm=1)
	univariate_model(d, 'time_delivery', 'target', normal=1, norm=1)
	univariate_polynomial_model(d, 'weight', 'target', polyfeat=2, alph=1e-7, normal=1, norm=1)
	univariate_polynomial_model(d, 'time_delivery', 'target', polyfeat=2, alph=1e-7, normal=1, norm=1)
	univariate_polynomial_model(d, 'prod_distance', 'target', polyfeat=2, alph=1e-7, normal=1, norm=1)
	univariate_polynomial_model(d, 'weight', 'target', polyfeat=3, alph=1e-11, th=[[0],[0],[0],[0]], normal=1, norm=1)
	univariate_polynomial_model(d, 'time_delivery', 'target', polyfeat=3, alph=1e-11, th=[[0],[0],[0],[0]], normal=1, norm=1)
	univariate_polynomial_model(d, 'prod_distance', 'target', polyfeat=3, alph=1e-11, th=[[0],[0],[0],[0]], normal=1, norm=1)
	univariate_polynomial_model(d, 'weight', 'target', polyfeat=4, alph=1e-11, th=[[0],[0],[0],[0],[0]], normal=1, norm=1)
	univariate_polynomial_model(d, 'time_delivery', 'target', polyfeat=4, alph=1e-11, th=[[0],[0],[0],[0],[0]], normal=1, norm=1)
	univariate_polynomial_model(d, 'prod_distance', 'target', polyfeat=4, alph=1e-11, th=[[0],[0],[0],[0],[0]], normal=1, norm=1)
	multivariate_model(d, ['prod_distance', 'time_delivery'], 'target', proportion=0.80, alph=1e-8, max_it=100000, th=[[0.0], [0.0], [0.0]], normal=1, norm=1)
	multivariate_model(d, ['prod_distance', 'weight'], 'target', proportion=0.80, alph=1e-8, max_it=100000, th=[[0.0], [0.0], [0.0]], normal=1, norm=1)
	multivariate_model(d, ['weight', 'time_delivery'], 'target', proportion=0.80, alph=1e-8, max_it=100000, th=[[0.0], [0.0], [0.0]], normal=1, norm=1)
	multivariate_model(d, ['prod_distance', 'weight', 'time_delivery'], 'target', proportion=0.80, alph=1e-8, max_it=100000, th=[[0.0], [0.0], [0.0], [0.0]], normal=1, norm=1)
	#multivariate_model(d, ['prod_distance', 'time_delivery', 'weight'], 'target', proportion=0.80, alph=1e-11, max_it=10000000, th=[[0.0], [0.0], [0.0], [0.0]], normal=0)

	# last try
	X = np.array(d[['prod_distance', 'weight', 'time_delivery']])
	Y = np.array(d[['target']])
	Xm = minmax(X)
	Ym = minmax(Y)
	x0 = add_polynomial_features(Xm[:,0], 2)
	x1 = add_polynomial_features(Xm[:,1], 2)
	x2 = add_polynomial_features(Xm[:,2], 2)
	Xl = np.c_[x0, x1, x2]
	clean_set = data_splitter(Xl, Ym, 0.80)
	Y_TRAIN = clean_set[2]
	X_TRAIN = clean_set[0]
	Y_TEST = clean_set[3]
	X_TEST = clean_set[1]
	myLR_f = MyLR([[0],[0],[0],[0],[0],[0],[0]])
	myLR_f.theta = myLR_f.normal_equation(X_TRAIN, Y_TRAIN)
	print(f"theta : {myLR_f.theta}")
	print(myLR_f.mse_(myLR_f.predict_(X_TEST), Y_TEST))

	X = np.array(d[['prod_distance', 'weight', 'time_delivery']])
	Y = np.array(d[['target']])
	Xm = minmax(X)
	Ym = minmax(Y)
	x0 = add_polynomial_features(Xm[:,0], 2)
	x1 = add_polynomial_features(Xm[:,1], 3)
	x2 = add_polynomial_features(Xm[:,2], 1)
	Xl = np.c_[x0, x1, x2]
	clean_set = data_splitter(Xl, Ym, 0.80)
	Y_TRAIN = clean_set[2]
	X_TRAIN = clean_set[0]
	Y_TEST = clean_set[3]
	X_TEST = clean_set[1]
	myLR_f = MyLR([[0],[0],[0],[0],[0],[0],[0]])
	myLR_f.theta = myLR_f.normal_equation(X_TRAIN, Y_TRAIN)
	print(f"theta : {myLR_f.theta}")
	print(myLR_f.mse_(myLR_f.predict_(X_TEST), Y_TEST))

	myLR_f.plot_uni(X_TEST[:,0], Y_TEST, myLR_f.predict_(X_TEST), "x1: weight", "y: target(minmax normalization)", "target", "predicted target")
	myLR_f.plot_uni(X_TEST[:,1], Y_TEST, myLR_f.predict_(X_TEST), "x1: prod_distance", "y: target(minmax normalization)", "target", "predicted target", coly='green', colyh='lightgreen')
	myLR_f.plot_uni(X_TEST[:,2], Y_TEST, myLR_f.predict_(X_TEST), "x1: time_delivery", "y: target(minmax normalization)", "target", "predicted target", coly='purple', colyh='#b19cd9')