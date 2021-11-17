import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression
from polynomial_model import add_polynomial_features

data = pd.read_csv('../are_blue_pills_magics.csv')
X = np.array(data[['Micrograms']])
Y = np.array(data[['Score']])

#MODEL1
X1 = add_polynomial_features(X, 1)
#model1 = MyLinearRegression([0, 0], alpha=0.0001, max_iter=400000)
model1 = MyLinearRegression([0, 0], alpha=0.00005, max_iter=100000)
model1.fit_(X1, Y)
YH1 = model1.predict_(X1)
M1 = model1.mse_(Y, YH1)
print(f"model1 MSE : {M1}")

#MODEL2
X2 = add_polynomial_features(X, 2)
#model1 = MyLinearRegression([0, 0], alpha=0.0001, max_iter=400000)
model2 = MyLinearRegression([0, 0, 0], alpha=0.00005, max_iter=100000)
model2.fit_(X2, Y)
YH2 = model2.predict_(X2)
M2 = (model2.mse_(Y, YH2))
print(f"model2 MSE : {M2}")

#MODEL2
X3 = add_polynomial_features(X, 3)
#model1 = MyLinearRegression([0, 0], alpha=0.0001, max_iter=400000)
model3 = MyLinearRegression([0, 0, 0, 0], alpha=0.00005, max_iter=100000)
model3.fit_(X3, Y)
YH3 = model3.predict_(X3)
M3 = (model3.mse_(Y, YH3))
print(f"model3 MSE : {M3}")

theta4 = np.array([-20, 160, -80, 10, -1]).reshape(-1,1)
#MODEL4
X4 = add_polynomial_features(X, 4)
model4 = MyLinearRegression(theta=theta4, alpha=2e-7, max_iter=100000)
model4.fit_(X4, Y)
YH4 = model4.predict_(X4)
M4 = (model4.mse_(Y, YH4))
print(f"model4 MSE : {M4}")

theta5 = np.array([1140, -1850, 1110, -305, 40, -2]).reshape(-1,1)
#MODEL5
X5 = add_polynomial_features(X, 5)
model5 = MyLinearRegression(theta=theta5, alpha=2e-8, max_iter=100000)
model5.fit_(X5, Y)
YH5 = model5.predict_(X5)
M5 = (model5.mse_(Y, YH5))
print(f"model5 MSE : {M5}")

theta6 = np.array([9110, -18015, 13400, -4935, 966, -96.4, 3.86]).reshape(-1,1)
#MODEL5
X6 = add_polynomial_features(X, 6)
model6 = MyLinearRegression(theta=theta6, alpha=1e-9, max_iter=100000)
model6.fit_(X6, Y)
YH6 = model6.predict_(X6)
M6 = model6.mse_(Y, YH6)
print(f"model6 MSE : {M6}")