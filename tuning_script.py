import numpy as np
from keras.utils import np_utils

# ---------------------------loading data-----------------------------
trainX = np.load("datasets/trainX_split1.npy")
trainX = trainX/255
trainX = trainX.reshape(trainX.shape[0], 48, 48, 1).astype('float32')

trainY = np.load("datasets/trainY_split1.npy")
trainY_oneHot = np_utils.to_categorical(trainY)

def get_class_weights(y):
    uni_val, counter = np.unique(y, return_counts=True)
    # print(uni_val)
    # print(counter)
    return  [float(10000/count) for count in counter]

class_weights=get_class_weights(trainY)

trainX_aug=np.load("datasets/trainX_split_zoom1.npy")
trainX_aug = trainX_aug.reshape(trainX_aug.shape[0], 48, 48, 1).astype('float32')


# ------------------------model construction--------------------------
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

model = Sequential()
# Create CN layer 1
model.add(Conv2D(filters=8,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(48, 48, 1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(filters=8,
                 kernel_size=(5,5),
                 padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
# Create Max-Pool 1
model.add(MaxPooling2D(pool_size=(2,2)))
# Add Dropout layer 1
model.add(Dropout(0.25))


# Create CN layer 2
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
# Create Max-Pool 2
model.add(MaxPooling2D(pool_size=(2,2)))
# Add Dropout layer 2
model.add(Dropout(0.25))

# Create CN layer 3
model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
# Create Max-Pool 3
model.add(MaxPooling2D(pool_size=(2,2)))
# Add Dropout layer 3
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

model.summary()
print("")


# ----------------------training phase------------------------
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# tempX = np.vstack((trainX, trainX_aug))
# tempY_oneHot = np.vstack((trainY_oneHot, trainY_oneHot_aug))
# print(tempX.shape, tempY.shape)
tempX = np.vstack((trainX, trainX_aug))
tempY_oneHot = np.vstack((trainY_oneHot, trainY_oneHot))
# print(tempX.shape, tempY_oneHot.shape)
train_history = model.fit(x=tempX, y=tempY_oneHot, validation_split=0.2, class_weight=class_weights,
                          epochs=10, batch_size=100, verbose=2)



















