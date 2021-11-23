from sys import prefix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from my_logistic_regression import MyLogisticRegression as MyLR
from minmax import minmax
from data_splitter_one_for_all import data_splitter_one_for_all_v2

data_training = pd.read_csv("./datasets/dataset_train.csv")
#print(data_training)
hand = pd.get_dummies(data_training['Best Hand'], prefix='hand')
#print(hand.head())
target = pd.get_dummies(data_training['Hogwarts House'], prefix='House')
print(target.head())
#final_target = data_training.iloc[:, 1:]
#print(final_target.head())
#clean_data = pd.concat()
clean_data = pd.concat([hand, data_training[['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']], target, data_training['Hogwarts House']], axis=1)
#print(clean_data)
#plt.figure(figsize=(10,7))
#plt.rcParams['figure.figsize']=(10,7)
g = sns.pairplot(data=clean_data[['hand_Left', 'hand_Right', 'Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying','Hogwarts House']], hue='Hogwarts House')
g.fig.set_figheight(7)
g.fig.set_figwidth(10)
plt.show()