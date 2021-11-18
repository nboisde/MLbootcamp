import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_splitter import data_splitter
from my_logistic_regression import MyLogisticRegression as MyLR
import sys

if __name__ == "__main__":
	args = sys.argv[1:]
	if len(args) != 1:
		quit("Too many arguments")
	s = args[0]
	s = s.split('=')
	if len(s) != 2:
		quit("Argument format error 1")
	if s[0] != '-zipcode':
		quit("Argument format error 2")
	try:
		zp = int(s[1])
	except:
		quit("zipcode must be a number")
	if not (zp >= 0 and zp <= 3):
		quit("zipcode must be between 0 and 3")
	print(zp)
	dssc = pd.read_csv("../solar_system_census.csv")
	print(dssc.head())
	dsscp = pd.read_csv("../solar_system_census_planets.csv")
	print(dsscp.head())
	dsscp = dsscp.iloc[: , 1:]
	new = pd.concat([dssc, dsscp], axis=1)
	print(new.head())
	new['Origin'] = np.where(new['Origin'] != float(zp), -1.0, new['Origin'])
	new['Origin'] = np.where(new['Origin'] == float(zp), 1.0, new['Origin'])
	new['Origin'] = np.where(new['Origin'] == -1.0, 0.0, new['Origin'])
	print(new.head())

	X = np.array(new[['height', 'weight', 'bone_density']])
	Y = np.array(new[['Origin']])
	#print(X)
	#print(Y)
	clean_set = data_splitter(X, Y, 0.80)
	Y_TRAIN = clean_set[2]
	X_TRAIN = clean_set[0]
	Y_TEST = clean_set[3]
	X_TEST = clean_set[1]

	model = MyLR([[0], [0], [0], [0]], max_iter=10000)
	#model.predict_(X_TRAIN)
	model.fit_(X_TRAIN, Y_TRAIN)
	print(model.theta)
	print(model.loss_(X_TRAIN, Y_TRAIN))
	print(model.predict_(X_TEST))

	model.plot_1D(X_TEST[:, 0], Y_TEST, model.predict_(X_TEST), xax="height", yax="planet", yleg="data", yhleg="predicted", coly='blue', colyh='lightblue')
	model.plot_1D(X_TEST[:, 1], Y_TEST, model.predict_(X_TEST), xax="weight", yax="planet", yleg="data", yhleg="predicted", coly='green', colyh='lightgreen')
	model.plot_1D(X_TEST[:, 2], Y_TEST, model.predict_(X_TEST), xax="bone_density", yax="planet", yleg="data", yhleg="predicted", coly='purple', colyh='#b19cd9')

	model.plot_3D(X_TEST, Y_TEST, model.predict_(X_TEST))