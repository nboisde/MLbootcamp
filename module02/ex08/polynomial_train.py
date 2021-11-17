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
model1 = MyLinearRegression([0, 0], alpha=0.0001, max_iter=45000)
model1.fit_(X1, Y)
YH1 = model1.predict_(X1)
print(model1.mse_(Y, YH1))

#MODEL2
X2 = add_polynomial_features(X, 2)
#model1 = MyLinearRegression([0, 0], alpha=0.0001, max_iter=400000)
model2 = MyLinearRegression([0, 0, 0], alpha=0.0001, max_iter=45000)
model2.fit_(X2, Y)
YH2 = model2.predict_(X2)
print(model2.mse_(Y, YH2))

#MODEL2
X3 = add_polynomial_features(X, 3)
#model1 = MyLinearRegression([0, 0], alpha=0.0001, max_iter=400000)
model3 = MyLinearRegression([0, 0, 0, 0], alpha=0.00005, max_iter=100000)
model3.fit_(X3, Y)
YH3 = model3.predict_(X3)
print(model3.mse_(Y, YH3))