%matplotlib inline

import os
# import matplotlib.pyplot as plt

# def plot_image(image, label=""):
#     fig = plt.gcf()
#     fig.set_size_inches(2,2)
#     plt.title("Label: {}".format(label))
#     plt.imshow(image, cmap='gray')
#     plt.show()

from keras.utils import np_utils
import numpy as np

trainY = np.load("datasets/trainY.npy")
print(trainY[:5])
trainY_oneHot = np_utils.to_categorical(trainY)

trainX=np.load("datasets/trainX.npy")
print(trainX.shape, trainX[0])

def get_class_weights(y):
    uni_val, counter = np.unique(y, return_counts=True)
    print(uni_val)
    print(counter)

    return  [float(10000/count) for count in counter]

class_weights=get_class_weights(trainY)
print(class_weights)

# for i in range(3):
#     plot_image(trainX[i].reshape(48, 48), trainY[i])


# trainY_oneHot_aug = np_utils.to_categorical(trainY_aug)
# print(trainY_oneHot_aug.shape)
trainX_aug1=np.load("datasets/trainX_hflip.npy")
print(trainX_aug1.shape, trainX_aug1[0])
# for i in range(3):
#     plot_image(trainX_aug1[i].reshape(48, 48), trainY[i])

trainX_aug2=np.load("datasets/trainX_bright.npy")
print(trainX_aug2.shape, trainX_aug2[0])
# for i in range(3):
#     plot_image(trainX_aug2[i].reshape(48, 48), trainY[i])

trainX_aug3=np.load("datasets/trainX_zoom.npy")
print(trainX_aug3.shape, trainX_aug3[0])
# for i in range(3):
#     plot_image(trainX_aug3[i].reshape(48, 48), trainY[i])

trainX_aug4=np.load("datasets/trainX_shear.npy")
print(trainX_aug4.shape, trainX_aug4[0])
# for i in range(3):
#     plot_image(trainX_aug4[i].reshape(48, 48), trainY[i])

trainX_aug5=np.load("datasets/trainX_rotate.npy")
print(trainX_aug5.shape, trainX_aug5[0])
# for i in range(3):
#     plot_image(trainX_aug5[i].reshape(48, 48), trainY[i])



trainX_aug1 = trainX_aug1.reshape(trainX_aug1.shape[0], 48, 48, 1).astype('float32')
trainX_aug2 = trainX_aug2.reshape(trainX_aug2.shape[0], 48, 48, 1).astype('float32')
trainX_aug3 = trainX_aug3.reshape(trainX_aug3.shape[0], 48, 48, 1).astype('float32')
trainX_aug4 = trainX_aug4.reshape(trainX_aug4.shape[0], 48, 48, 1).astype('float32')
trainX_aug5 = trainX_aug5.reshape(trainX_aug5.shape[0], 48, 48, 1).astype('float32')
trainX = trainX.reshape(trainX.shape[0], 48, 48, 1).astype('float32')


import h5py
from keras.models import load_model, save_model
model = load_model('models/09211011_dataAugAll_model5.h5')


tempX = np.vstack((trainX, trainX_aug1, trainX_aug2, trainX_aug3, trainX_aug4, trainX_aug5))
tempY_oneHot = np.vstack((trainY_oneHot, trainY_oneHot, trainY_oneHot, trainY_oneHot, trainY_oneHot, trainY_oneHot))
print(tempX.shape, tempY_oneHot.shape)
train_history = model.fit(x=tempX, y=tempY_oneHot, validation_split=0.2, class_weight=class_weights,
                          epochs=50, batch_size=100, verbose=2)

model.save('models/092X_dataAugAll_model.h5')

