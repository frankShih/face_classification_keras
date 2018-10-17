# -------------------- load raw data -------------------
import numpy as np

label=[]
raw_image=[]
'''
for line in open('./Homework/train.csv'):
    line = line.split(',')
    if not(line[0].isdigit()):
        continue
    label.append(line[0])
    raw_image.append(line[1].split())

temp = np.array(raw_image).reshape(len(raw_image), 48, 48).astype(np.float32)
temp = temp/255
np.save("trainX", temp)
temp = np.array(label).reshape(len(label), 1).astype(np.int)
np.save("trainY", temp)


for line in open('./Homework/test.csv'):
    line = line.split(',')
    if not(line[0].isdigit()):
        continue
    label.append(line[0])
    raw_image.append(line[1].split())

temp = np.array(raw_image).reshape(len(raw_image), 48, 48).astype(np.float32)
temp = temp/255
np.save("testX", temp)
'''



# ---------------- loading format data -----------------

import numpy as np
from keras.utils import np_utils


trainX = np.load("datasets/trainX.npy").astype('float32')
# print(type(trainX[0]), trainX.shape)
# trainX = trainX/255
# print(trainX[0])
# np.save("datasets/trainX_bright1.npy", trainX)

trainY = np.load("datasets/trainY.npy")
# print(type(trainY[0]), trainY.shape)
# print(trainY[:5])

trainX = trainX.reshape(trainX.shape[0], 48, 48, 1).astype('float32')
trainY_oneHot = np_utils.to_categorical(trainY)
# print(trainY_oneHot.shape)


def get_class_weights(y):
    uni_val, counter = np.unique(y, return_counts=True)
    # print(uni_val)
    # print(counter)
    return  [float(10000/count) for count in counter]

class_weights=get_class_weights(trainY)
print(class_weights)


# -----------------------stratified sampling --------------------------

from sklearn.model_selection import StratifiedShuffleSplit

def stratified_sampling(data, label, valid_percent=0.2):
    spliter = StratifiedShuffleSplit(n_splits=int(1/valid_percent), test_size=valid_percent, train_size=1-valid_percent, random_state=0)
    train_index_set, valid_index_set = [], []
    for train_index, valid_index in spliter.split(data, label):
        print("TRAIN:", train_index, len(train_index), "TEST:", valid_index, len(valid_index))
        train_index_set.append(train_index)
        valid_index_set.append(valid_index)
    return train_index_set, valid_index_set

t_ind, v_ind = stratified_sampling(trainX, trainY, valid_percent=0.2)

counter=0
for tt, vv in zip(t_ind, v_ind):
    counter+=1
    np.save("trainX_split{}".format(counter), trainX[tt])
    np.save("validX_split{}".format(counter), trainX[vv])
    np.save("trainY_split{}".format(counter), trainY[tt])
    np.save("validY_split{}".format(counter), trainY[vv])

