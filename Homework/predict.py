import numpy as np

for line in open('./Homework/test.csv'):
    line = line.split(',')
    if not(line[0].isdigit()):
        continue
    label.append(line[0])   # id, in test set
    raw_image.append(line[1].split())

testX = np.array(raw_image).reshape(len(raw_image),48,48).astype(np.int)
testX = testX/255
print(testX.shape, temp[0])



# ------------------------ model predicting --------------------------
model = load_model('model_9.h5')   # load the best model during training
testX = testX.reshape(testX.shape[0], 48, 48, 1).astype('float32')
prediction_prob = model.predict(testX, batch_size=None, verbose=0, steps=None)
prediction = np.argmax(prediction_prob, axis=1)
print(prediction.shape, prediction_prob[:5], prediction[:5])
np.savetxt("prediction.csv", prediction, delimiter=",")

