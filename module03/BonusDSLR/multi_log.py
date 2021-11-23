from sys import prefix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_splitter_one_for_all import data_splitter_one_for_all, data_splitter_one_for_all_v2
from my_logistic_regression import MyLogisticRegression as MyLR
from minmax import minmax

dssc = pd.read_csv("../solar_system_census.csv")
print(dssc.head())
dsscp = pd.read_csv("../solar_system_census_planets.csv")
print(dsscp.head())
ori = dsscp.iloc[: , 1:]
dsscp = pd.get_dummies(dsscp['Origin'], prefix='planet')
#print(dsscp.head())
new = pd.concat([dssc, dsscp, ori], axis=1)
print(new.head())

X = np.array(new[['height', 'weight', 'bone_density']])
Y = np.array(new[['planet_0.0', 'planet_1.0', 'planet_2.0', 'planet_3.0']])
C = np.array(new)
print(C)
tpl = data_splitter_one_for_all(X, Y)
tpl2 = data_splitter_one_for_all_v2(C)
print(np.c_[tpl2[0], tpl2[2], tpl2[4]])
print(np.c_[tpl2[1], tpl2[3], tpl2[5]])
#print(tpl[0], tpl[2])
#print(tpl[1], tpl[3])

X_TRAIN = np.c_[minmax(tpl2[2][:,0]),minmax(tpl2[2][:,1]),tpl2[2][:,2]]
X_TEST = np.c_[minmax(tpl2[3][:,0]),minmax(tpl2[3][:,1]),tpl2[3][:,2]]

#plt.hist(X_TRAIN)
#plt.show()
# THIS PIECE OF CODE SHOWS THAT DATA SET CANNOT ESTABLISH ANY TENDENCY WITH THE DATA BECAUSE DISTRIBUTION OF DATA ARE NOT ESPACED CONSIDERING ONE ANOTHER...
new["height x weight"] = new["height"] * new["weight"]
new["height x bone"] = new["height"] * new["bone_density"]
new["bone x weight"] = new["bone_density"] * new["weight"]
new["w x h x h"] = new["weight"] * new["height"] * new["height"]
sns.pairplot(data=new[["height", "weight", "bone_density", "height x weight", "height x bone", "bone x weight", "w x h x h", "Origin"]],hue="Origin")
plt.show()

print(X_TRAIN)

Y_TRAIN0 = (tpl2[4][:, 0]).reshape(-1, 1)
Y_TRAIN1 = tpl2[4][:, 1].reshape(-1, 1)
Y_TRAIN2 = tpl2[4][:, 2].reshape(-1, 1)
Y_TRAIN3 = tpl2[4][:, 3].reshape(-1, 1)
#print(Y_TRAIN0)

Y_TEST0 = tpl2[5][:, 0].reshape(-1, 1)
Y_TEST1 = tpl2[5][:, 1].reshape(-1, 1)
Y_TEST2 = tpl2[5][:, 2].reshape(-1, 1)
Y_TEST3 = tpl2[5][:, 3].reshape(-1, 1)

def model_training_3(xtr, ytr, xts, yts, th=[[0], [0], [0], [0]], model_name="logistic regression on one feature", al=0.0001, mi=100000):
	print("--------------------------------------------------")
	print(f"{model_name}")
	print(f"caracteristics: alpha: {al}, max_iteration: {mi}")
	print("--------------------------------------------------")
	model = MyLR([[0], [0], [0], [0]], alpha=al, max_iter=mi)
	model.fit_(xtr, ytr)
	print("model theta after fitting:")
	print(model.theta)
	print("model loss on training data: ", model.loss_(xtr, ytr))
	y_pred = model.predict_(xts)
	y_predtr = model.predict_(xtr)
	print("model loss on test data: ", model.loss_(xts, yts))
	return (model.theta, y_pred, y_predtr)

print("DATA NORMALIZED")
# RUN TRAINING
Y_PRED0 = model_training_3(X_TRAIN, Y_TRAIN0, X_TEST, Y_TEST0, model_name="logistic regression for planet 0", al=0.001, mi=1000000)
#print(np.c_[tpl2[1], Y_PRED0[1]])

# AVOID RUN TIME WITH A GIVEN THETA USING TRAINING ABOVE
#theta0 = [[ 0.2050242 ], [-0.00209291], [-0.01480952], [ 0.41265199]]
#model0 = MyLR(theta0, alpha=0.0001, max_iter=200000)
#Y_PRED0 = (model0.theta, model0.predict_(X_TEST), model0.predict_(X_TRAIN))
#print(np.c_[tpl2[1], Y_PRED0[1]])
#print(X_TRAIN)


# TRAINING
Y_PRED1 = model_training_3(X_TRAIN, Y_TRAIN1, X_TEST, Y_TEST1, model_name="logistic regression for planet 1", al=0.001, mi=1000000)
#print(np.c_[tpl2[1], Y_PRED1[1]])

# AVOID
#theta1 = [[ 0.13666136], [-0.01942118], [ 0.0197048 ], [ 0.68422153]]
#model1 = MyLR(theta1, alpha=0.0001, max_iter=200000)
#Y_PRED1 = (model1.theta, model1.predict_(X_TEST), model1.predict_(X_TRAIN))
#print(np.c_[tpl2[1], Y_PRED1[1]])

# TRAINING
Y_PRED2 = model_training_3(X_TRAIN, Y_TRAIN2, X_TEST, Y_TEST2, model_name="logistic regression for planet 2", al=0.001, mi=1000000)
#print(np.c_[tpl2[1], Y_PRED2[1]])

# AVOID
#theta2 = [[-0.2666801 ],[-0.04619969],[ 0.09571756],[-0.41364565]]
#model2 = MyLR(theta2, alpha=0.0001, max_iter=200000)
#Y_PRED2 = (model2.theta, model2.predict_(X_TEST), model2.predict_(X_TRAIN))
#print(np.c_[tpl2[1], Y_PRED2[1]])

# TRAINING
Y_PRED3 = model_training_3(X_TRAIN, Y_TRAIN3, X_TEST, Y_TEST3, model_name="logistic regression for planet 3", al=0.001, mi=100000)
#print(np.c_[tpl2[1], Y_PRED3[1]])

# AVOID
#0.0001 200000
#theta3 = [[-0.18165991],[ 0.03715523],[-0.09023838],[-0.82388675]]
#0.0001 500000
#theta3 = [[-0.55319984],[-0.05210611],[ 0.11292212],[-0.68820559]]
#model3 = MyLR(theta3, alpha=0.0001, max_iter=200000)
#Y_PRED3 = (model3.theta, model3.predict_(X_TEST), model3.predict_(X_TRAIN))
#LOSS3 = model3.loss_(X_TRAIN, Y_TRAIN3)
#print(LOSS3)
#print(np.c_[tpl2[1], Y_PRED3[1]])

ALL_PREDS = np.c_[tpl2[1], Y_PRED0[1], Y_PRED1[1], Y_PRED2[1], Y_PRED3[1], tpl2[7]]
print(ALL_PREDS)

#np.argmax our np.argmin a appliquer sur ALL_PREDS pour trouver l'index...