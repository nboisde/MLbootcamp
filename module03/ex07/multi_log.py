from sys import prefix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_splitter_one_for_all import data_splitter_one_for_all
from my_logistic_regression import MyLogisticRegression as MyLR

dssc = pd.read_csv("../solar_system_census.csv")
print(dssc.head())
dsscp = pd.read_csv("../solar_system_census_planets.csv")
print(dsscp.head())
dsscp = pd.get_dummies(dsscp['Origin'], prefix='planet')
print(dsscp.head())
new = pd.concat([dssc, dsscp], axis=1)
print(new.head())

X = np.array(new[['height', 'weight', 'bone_density']])
Y = np.array(new[['planet_0.0', 'planet_1.0', 'planet_2.0', 'planet_3.0']])

tpl = data_splitter_one_for_all(X, Y)
print(tpl[0], tpl[2])
print(tpl[1], tpl[3])