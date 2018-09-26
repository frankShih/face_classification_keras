label=[]
raw_image=[]

# --------------------- format data ----------------------
for line in open('./train.csv'):
    line = line.split(',')
    if not(line[0].isdigit()):
        continue
    label.append(line[0]) #label, in train set
    raw_image.append(line[1].split())

import numpy as np
from keras.utils import np_utils
trainX = np.array(raw_image).reshape(len(raw_image),48,48).astype(np.int)
trainX = trainX/255
# print(trainX.shape, trainX[0])

trainY = np.array(label).reshape(len(label),1).astype(np.int)
# print(trainY.shape)
trainY_oneHot = np_utils.to_categorical(trainY)
# print(trainY_oneHot.shape)

# for line in open('./Homework/test.csv'):
#     line = line.split(',')
#     if not(line[0].isdigit()):
#         continue
#     label.append(line[0])   # id, in test set
#     raw_image.append(line[1].split())

# testX = np.array(raw_image).reshape(len(raw_image),48,48).astype(np.int)
# testX = testX/255
# print(testX.shape, temp[0])

# -------------------- get_class_weights for imbalance dataset --------------------
def get_class_weights(y):
    uni_val, counter = np.unique(y, return_counts=True)
    # print(uni_val)
    # print(counter)
    return  [float(10000/count) for count in counter]

class_weights=get_class_weights(trainY)
print(class_weights)


# ---------------------- image data augmentation --------------------------
trainX_hflip = trainX[:, :, ::-1]  # horizontal, vertical for axis=1
trainX_hflip = np.concatenate(trainX_hflip, axis=0).reshape(trainX.shape[0], 48, 48, 1).astype('float32')  
print('hflip shape:{}, {}'.format(trainX_hflip.shape, trainX_hflip[0]))

trainX = trainX.reshape(trainX.shape[0], 48, 48, 1).astype('float32')
print('train_x shape:{}, {}'.format(trainX.shape, trainX[0]))


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def data_augment(trainX, trainY, imageGenerator):
    imageGenerator.fit(trainX)
    counter=0
    trainX_aug = []
    trainY_aug = []
    generate_amount=len(trainX)

    for bx, by in imageGenerator.flow(trainX, trainY, batch_size=1, shuffle=False):
        counter+=1
        if counter>generate_amount: break

        bx=np.squeeze(bx, axis=3)
        trainX_aug.append(bx)
        trainY_aug.append(by)

    trainX_aug = np.concatenate(trainX_aug, axis=0).reshape(generate_amount, 48, 48)
#    print(trainX_aug.shape, trainX_aug[0])

    return trainX_aug


datagen = ImageDataGenerator(fill_mode='constant', rotation_range=20)
trainX_rotate = data_augment(trainX, trainY, datagen)
trainX_rotate = trainX_rotate.reshape(trainX_rotate.shape[0], 48, 48, 1).astype('float32') 
print('rotate shape:{}, {}'.format(trainX_rotate.shape, trainX_rotate[0]))
datagen = ImageDataGenerator(fill_mode='constant', shear_range=20)
trainX_shear = data_augment(trainX, trainY, datagen)
trainX_shear = trainX_shear.reshape(trainX_shear.shape[0], 48, 48, 1).astype('float32') 
print('shear shape:{}, {}'.format(trainX_shear.shape, trainX_shear[0]))
datagen = ImageDataGenerator(fill_mode='constant', zoom_range=0.2)
trainX_zoom = data_augment(trainX, trainY, datagen)
trainX_zoom = trainX_zoom.reshape(trainX_zoom.shape[0], 48, 48, 1).astype('float32') 
print('zoom shape:{}, {}'.format(trainX_zoom.shape, trainX_zoom[0]))
datagen = ImageDataGenerator(fill_mode='constant', brightness_range=[0.8, 1.2])
trainX_bright = data_augment(trainX, trainY, datagen)
trainX_bright = trainX_bright.reshape(trainX_bright.shape[0], 48, 48, 1).astype('float32')/255 
print('bright shape:{}, {}'.format(trainX_bright.shape, trainX_bright[0]))


# ---------------------- model building --------------------------
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers

model = Sequential()
# Create CN layer 1
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', input_shape=(48, 48, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Create CN layer 2
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Create CN layer 3
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.summary()
print("")


# ------------------------ model training --------------------------
import h5py
from keras.models import load_model, save_model
from keras import optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tempX = np.vstack((trainX, trainX_rotate, trainX_shear, trainX_zoom, trainX_bright, trainX_hflip))
tempY_oneHot = np.vstack((trainY_oneHot, trainY_oneHot, trainY_oneHot, trainY_oneHot, trainY_oneHot, trainY_oneHot))
print(tempX.shape, tempY_oneHot.shape)


for i in range(10):
    train_history = model.fit(x=tempX, y=tempY_oneHot, validation_split=0.2, class_weight=class_weights,
                          epochs=1, batch_size=100, verbose=2)
    model.save('model_{}_temp.h5'.format(i))
# PS. I do not implement any early-stopping criteria, hence I save the model every 10 epoch



