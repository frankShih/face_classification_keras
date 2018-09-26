import numpy as np
import pandas as pd

label = []
raw_image=[]

for line in open('./test.csv'):
    line = line.split(',')
    if not(line[0].isdigit()):
        continue
    label.append(line[0])   # id, in test set
    raw_image.append(line[1].split())

testX = np.array(raw_image).reshape(len(raw_image),48,48).astype(np.int)
testX = testX/255
print(testX.shape, testX[0])


import h5py
from keras.models import load_model, save_model

# ------------------------ model predicting --------------------------
model = load_model('../models/dataAugAll_1_model_2.h5')   # load the best model during training
testX = testX.reshape(testX.shape[0], 48, 48, 1).astype('float32')
prediction_prob = model.predict(testX, batch_size=None, verbose=0, steps=None)
prediction = np.argmax(prediction_prob, axis=1)
print(prediction.shape, prediction_prob[:5], prediction[:5], label[:5])
result = pd.DataFrame(np.vstack((np.array(label), prediction)).T, columns=['id', 'label'])
# np.savetxt("prediction.csv", prediction, delimiter=",", header='id')

result.to_csv("prediction1.csv", sep=',', index=False)