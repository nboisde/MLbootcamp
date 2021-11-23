from sys import prefix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_splitter_one_for_all import data_splitter_one_for_all_v2
#from module03.BonusDSLR.data_splitter_one_for_all import data_splitter_one_for_all_dslr
from my_logistic_regression import MyLogisticRegression as MyLR
from minmax import minmax
from data_splitter_one_for_all import data_splitter_one_for_all

data_training = pd.read_csv("./datasets/dataset_train.csv")
#hand = pd.get_dummies(data_training['Best Hand'], prefix='hand')
target = pd.get_dummies(data_training['Hogwarts House'], prefix='House')
print(target)
clean_data = pd.concat([data_training[['Index','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']], target, data_training['Hogwarts House']], axis=1)
#print(clean_data)
#g = sns.pairplot(data=clean_data[['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying','Hogwarts House']], hue='Hogwarts House')
#g.fig.set_figheight(7)
#g.fig.set_figwidth(10)
#plt.show()


#train_set = pd.concat([clean_data[['Index','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Care of Magical Creatures','Charms','Flying','Transfiguration']], target], axis=1)
#print(train_set.head())
#train_set.dropna(subset = ['Index','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Care of Magical Creatures','Charms','Flying','Transfiguration'], inplace=True)
#print(train_set.head())
#
#C = np.array(train_set)
##print(C)
#tpl = data_splitter_one_for_all_dslr(C, xfeat=8, proportion=0.80)
#
#
#X_TRAIN = tpl[2]
#X_TEST = tpl[3]
#INDEX_TRAIN = tpl[0]
#print(INDEX_TRAIN)
#INDEX_TEST = tpl[1]
#
## One for all training
#Y_GRIF_TR = tpl[4][:, 0].reshape(-1, 1)
#Y_HUF_TR = tpl[4][:, 1].reshape(-1, 1)
#Y_RAVEN_TR = tpl[4][:, 2].reshape(-1, 1)
#Y_SLYT_TR = tpl[4][:, 3].reshape(-1, 1)
#
#Y_TEST_GRIF = tpl[5][:, 0].reshape(-1, 1)
#Y_TEST_HUF = tpl[5][:, 1].reshape(-1, 1)
#Y_TEST_RAVEN = tpl[5][:, 2].reshape(-1, 1)
#Y_TEST_SLYT = tpl[5][:, 3].reshape(-1, 1)
#
#trg = np.c_[INDEX_TRAIN, X_TRAIN, Y_GRIF_TR]
#trs = np.c_[INDEX_TRAIN, X_TRAIN, Y_SLYT_TR]
#print(trs)
#tsg = np.c_[INDEX_TEST, X_TEST, Y_TEST_GRIF]
#print(trg)
#print(tsg)


def model_training(xtr, ytr, xts, yts, th=[[0], [0], [0], [0], [0], [0], [0], [0], [0]], name="log reg on house...", al=0.0001, mi=10000):
	print("--------------------------------------------------")
	print(f"{name}")
	print(f"caracteristics: alpha: {al}, max_iteration: {mi}")
	print("--------------------------------------------------")
	model = MyLR(th, alpha=al, max_iter=mi)
	model.fit_(xtr, ytr)
	print("model theta after fitting:")
	print(model.theta)
	print("model loss on training data: ", model.loss_(xtr, ytr))
	y_pred = model.predict_(xts)
	y_predtr = model.predict_(xtr)
	#print("model loss on test data: ", model.loss_(xts, yts))
	return (model.theta, y_pred, y_predtr)

#MGRIF = model_training(X_TRAIN, Y_GRIF_TR, X_TEST, Y_TEST_GRIF, al=0.0001, mi=1000)
#MHUF = model_training(X_TRAIN, Y_HUF_TR, X_TEST, Y_TEST_HUF, al=0.0001, mi=1000)
#MSLYT = model_training(X_TRAIN, Y_SLYT_TR, X_TEST, Y_TEST_SLYT, al=0.0001, mi=1000)
#MRAVEN = model_training(X_TRAIN, Y_RAVEN_TR, X_TEST, Y_TEST_RAVEN, al=0.0001, mi=1000)

train_set = pd.concat([clean_data[['Index','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Care of Magical Creatures','Charms','Flying','Transfiguration']], target, clean_data['Hogwarts House']], axis=1)
#print(train_set.head())
train_set.dropna(subset = ['Index','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Care of Magical Creatures','Charms','Flying','Transfiguration'], inplace=True)
print(train_set)

def data_splitter_one_for_all(d, xfeat=8, yclass=4, proportion=0.80):
    if not isinstance(d, np.ndarray):
        return None
    if not isinstance(proportion, float):
        return None
    if proportion > 1 or proportion < 0:
        return None
    rng = np.random.default_rng()
    rng.shuffle(d)
    split_horizontally_idx = int(d.shape[0] * proportion)
    train = d[:split_horizontally_idx , :]
    test = d[split_horizontally_idx: , :]
    id_train = train
    id_test = test
    for i in range(yclass + xfeat + 1):
        id_train = np.delete(id_train, -1, axis=1)
        id_test = np.delete(id_test, -1, axis=1)
    x_train = train
    x_test = test
    for i in range(yclass + 1):
        x_train = np.delete(x_train, -1, axis=1)
        x_test = np.delete(x_test, -1, axis=1)
    x_train = np.delete(x_train, 0, axis=1)
    x_test = np.delete(x_test, 0, axis=1)
    y_train = train
    y_test = test
    for i in range(xfeat + 1):
        y_train = np.delete(y_train, 0, axis=1)
        y_test = np.delete(y_test, 0, axis=1)
    y_train = np.delete(y_train, -1, axis=1)
    y_test = np.delete(y_test, -1, axis=1)
    real_train = train[:, -1].reshape(-1, 1)
    real_test = test[:, -1].reshape(-1, 1)
    return (id_train, id_test, x_train, x_test, y_train, y_test, real_train, real_test)

C = np.array(train_set)
#print(C)
tpl = data_splitter_one_for_all(C)

#X_TRAIN = tpl[2]
#X_TEST = tpl[3]
X_TRAIN = np.c_[minmax(tpl[2][:,0]),minmax(tpl[2][:,1]),minmax(tpl[2][:,2]),minmax(tpl[2][:,3]),minmax(tpl[2][:,4]),minmax(tpl[2][:,5]),minmax(tpl[2][:,6]),minmax(tpl[2][:,7])]
X_TEST = np.c_[minmax(tpl[3][:,0]),minmax(tpl[3][:,1]),minmax(tpl[3][:,2]),minmax(tpl[3][:,3]),minmax(tpl[3][:,4]),minmax(tpl[3][:,5]),minmax(tpl[3][:,6]),minmax(tpl[3][:,7])]
INDEX_TRAIN = tpl[0]
INDEX_TEST = tpl[1]
print(tpl[0])
print(tpl[2])
print(tpl[4])


print(np.c_[tpl[0], tpl[2], tpl[4], tpl[6]])
#print(tpl[4])

# One for all training
Y_GRIF_TR = tpl[4][:, 0].reshape(-1, 1)
Y_HUF_TR = tpl[4][:, 1].reshape(-1, 1)
Y_RAVEN_TR = tpl[4][:, 2].reshape(-1, 1)
Y_SLYT_TR = tpl[4][:, 3].reshape(-1, 1)

Y_TEST_GRIF = tpl[5][:, 0].reshape(-1, 1)
Y_TEST_HUF = tpl[5][:, 1].reshape(-1, 1)
Y_TEST_RAVEN = tpl[5][:, 2].reshape(-1, 1)
Y_TEST_SLYT = tpl[5][:, 3].reshape(-1, 1)


MGRIF = model_training(X_TRAIN, Y_GRIF_TR, X_TEST, Y_TEST_GRIF,th=[[0], [0], [0], [0], [0], [0], [0], [0], [0]],name='Gryffindor', al=0.2, mi=100)
MHUF = model_training(X_TRAIN, Y_HUF_TR, X_TEST, Y_TEST_HUF,th=[[0], [0], [0], [0], [0], [0], [0], [0], [0]],name='Hufflepuf', al=0.2, mi=100)
MSLYT = model_training(X_TRAIN, Y_SLYT_TR, X_TEST, Y_TEST_SLYT,th=[[0], [0], [0], [0], [0], [0], [0], [0], [0]],name='Slytherin', al=0.2, mi=100)
MRAVEN = model_training(X_TRAIN, Y_RAVEN_TR, X_TEST, Y_TEST_RAVEN,th=[[0], [0], [0], [0], [0], [0], [0], [0], [0]],name='Ravenclaw', al=0.2, mi=100)

ALL_PREDS = np.c_[tpl[1], MGRIF[1], MHUF[1], MRAVEN[1], MSLYT[1], tpl[7]]
print(ALL_PREDS)

DF = pd.DataFrame(ALL_PREDS, columns=['Index', 'Gryffindor Probability', 'Hufflepuff Probability', 'Ravenclaw Probability', 'Slytherin Probability', 'Real House'])
print(DF)
DF.to_csv(path_or_buf='predict.csv', sep=',',index=False)

train_set = pd.concat([clean_data[['Index','Divination','Muggle Studies','Care of Magical Creatures','Charms','Flying','Transfiguration']], target, clean_data['Hogwarts House']], axis=1)
#print(train_set.head())
train_set.dropna(subset = ['Index','Divination','Muggle Studies','Care of Magical Creatures','Charms','Flying','Transfiguration'], inplace=True)
print(train_set)
#g = sns.pairplot(data=clean_data[['Divination','Muggle Studies','Transfiguration','Care of Magical Creatures','Charms','Flying','Hogwarts House']], hue='Hogwarts House')
#g.fig.set_figheight(7)
#g.fig.set_figwidth(10)
#plt.show()

C = np.array(train_set)
#print(C)
tpl = data_splitter_one_for_all(C, xfeat=6)

#X_TRAIN = tpl[2]
#X_TEST = tpl[3]
X_TRAIN = np.c_[minmax(tpl[2][:,0]),minmax(tpl[2][:,1]),minmax(tpl[2][:,2]),minmax(tpl[2][:,3]),minmax(tpl[2][:,4]),minmax(tpl[2][:,5])]
X_TEST = np.c_[minmax(tpl[3][:,0]),minmax(tpl[3][:,1]),minmax(tpl[3][:,2]),minmax(tpl[3][:,3]),minmax(tpl[3][:,4]),minmax(tpl[3][:,5])]
INDEX_TRAIN = tpl[0]
INDEX_TEST = tpl[1]
print(tpl[0])
print(tpl[2])
print(tpl[4])
print(np.c_[tpl[0], tpl[2], tpl[4], tpl[6]])


#print(np.c_[tpl[0], tpl[2], tpl[4], tpl[6]])
#print(tpl[4])

# One for all training
Y_GRIF_TR = tpl[4][:, 0].reshape(-1, 1)
Y_HUF_TR = tpl[4][:, 1].reshape(-1, 1)
Y_RAVEN_TR = tpl[4][:, 2].reshape(-1, 1)
Y_SLYT_TR = tpl[4][:, 3].reshape(-1, 1)

Y_TEST_GRIF = tpl[5][:, 0].reshape(-1, 1)
Y_TEST_HUF = tpl[5][:, 1].reshape(-1, 1)
Y_TEST_RAVEN = tpl[5][:, 2].reshape(-1, 1)
Y_TEST_SLYT = tpl[5][:, 3].reshape(-1, 1)


MGRIF = model_training(X_TRAIN, Y_GRIF_TR, X_TEST, Y_TEST_GRIF,th=[[0], [0], [0], [0], [0], [0], [0]],name='Gryffindor', al=0.2, mi=100)
MHUF = model_training(X_TRAIN, Y_HUF_TR, X_TEST, Y_TEST_HUF,th=[[0], [0], [0], [0], [0], [0], [0]],name='Hufflepuf', al=0.2, mi=100)
MSLYT = model_training(X_TRAIN, Y_SLYT_TR, X_TEST, Y_TEST_SLYT,th=[[0], [0], [0], [0], [0], [0], [0]],name='Slytherin', al=0.2, mi=100)
MRAVEN = model_training(X_TRAIN, Y_RAVEN_TR, X_TEST, Y_TEST_RAVEN,th=[[0], [0], [0], [0], [0], [0], [0]],name='Ravenclaw', al=0.2, mi=100)

ALL_PREDS = np.c_[tpl[1], MGRIF[1], MHUF[1], MRAVEN[1], MSLYT[1], tpl[7]]
print(ALL_PREDS)

DF = pd.DataFrame(ALL_PREDS, columns=['Index', 'Gryffindor Probability', 'Hufflepuff Probability', 'Ravenclaw Probability', 'Slytherin Probability', 'Real House'])
print(DF)

train_set = pd.concat([clean_data[['Index','Divination','Muggle Studies','Care of Magical Creatures','Potions','Charms','Flying','Transfiguration']], target, clean_data['Hogwarts House']], axis=1)
#print(train_set.head())
train_set.dropna(subset = ['Index','Divination','Muggle Studies','Care of Magical Creatures','Potions','Charms','Flying','Transfiguration'], inplace=True)
print(train_set)

C = np.array(train_set)
#print(C)
tpl = data_splitter_one_for_all(C, xfeat=7)

#X_TRAIN = tpl[2]
#X_TEST = tpl[3]
X_TRAIN = np.c_[minmax(tpl[2][:,0]),minmax(tpl[2][:,1]),minmax(tpl[2][:,2]),minmax(tpl[2][:,3]),minmax(tpl[2][:,4]),minmax(tpl[2][:,5]),minmax(tpl[2][:,6])]
X_TEST = np.c_[minmax(tpl[3][:,0]),minmax(tpl[3][:,1]),minmax(tpl[3][:,2]),minmax(tpl[3][:,3]),minmax(tpl[3][:,4]),minmax(tpl[3][:,5]),minmax(tpl[3][:,6])]
INDEX_TRAIN = tpl[0]
INDEX_TEST = tpl[1]
#print(tpl[0])
#print(tpl[2])
#print(tpl[4])
print(np.c_[tpl[0], tpl[2], tpl[4], tpl[6]])


#print(np.c_[tpl[0], tpl[2], tpl[4], tpl[6]])
#print(tpl[4])

# One for all training
Y_GRIF_TR = tpl[4][:, 0].reshape(-1, 1)
Y_HUF_TR = tpl[4][:, 1].reshape(-1, 1)
Y_RAVEN_TR = tpl[4][:, 2].reshape(-1, 1)
Y_SLYT_TR = tpl[4][:, 3].reshape(-1, 1)

Y_TEST_GRIF = tpl[5][:, 0].reshape(-1, 1)
Y_TEST_HUF = tpl[5][:, 1].reshape(-1, 1)
Y_TEST_RAVEN = tpl[5][:, 2].reshape(-1, 1)
Y_TEST_SLYT = tpl[5][:, 3].reshape(-1, 1)


MGRIF = model_training(X_TRAIN, Y_GRIF_TR, X_TEST, Y_TEST_GRIF,th=[[0], [0], [0], [0], [0], [0], [0], [0]],name='Gryffindor', al=0.2, mi=100)
MHUF = model_training(X_TRAIN, Y_HUF_TR, X_TEST, Y_TEST_HUF,th=[[0], [0], [0], [0], [0], [0], [0], [0]],name='Hufflepuf', al=0.2, mi=100)
MSLYT = model_training(X_TRAIN, Y_SLYT_TR, X_TEST, Y_TEST_SLYT,th=[[0], [0], [0], [0], [0], [0], [0], [0]],name='Slytherin', al=0.2, mi=100)
MRAVEN = model_training(X_TRAIN, Y_RAVEN_TR, X_TEST, Y_TEST_RAVEN,th=[[0], [0], [0], [0], [0], [0], [0], [0]],name='Ravenclaw', al=0.2, mi=100)

ALL_PREDS = np.c_[tpl[1], MGRIF[1], MHUF[1], MRAVEN[1], MSLYT[1], tpl[7]]
print(ALL_PREDS)

DF = pd.DataFrame(ALL_PREDS, columns=['Index', 'Gryffindor Probability', 'Hufflepuff Probability', 'Ravenclaw Probability', 'Slytherin Probability', 'Real House'])
print(DF)
#DF.to_csv(path_or_buf='predict.csv', sep=',',index=False)