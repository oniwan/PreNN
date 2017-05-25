
# coding: utf-8

# In[ ]:

import keras
from keras.models import Sequential  
from keras.layers.core import Dense, Activation,LSTM,Bidirectional
from keras.callbacks import EarlyStopping, CSVLogger
import math    


# In[ ]:

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


# In[ ]:

dataframe = pd.read_csv("./TrainingSet.csv")
print dataframe.shape
dataframe = dataframe.dropna()
print dataframe.shape
dataframe.head()


# In[ ]:

dataframe.drop(["YYYYMM"],axis=1,inplace=True)
dataset = dataframe.values
print dataset
dataset = dataset.astype("float32")
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print "-"*20
print dataset


# In[ ]:

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# In[ ]:

def split_xy(dataset):
    dataX = dataset[:,1:]
    dataY = dataset[:,0]
    return np.array(dataX),np.array([dataY]).T
trainX, trainY = split_xy(train)
testX,testY = split_xy(test)


# In[ ]:

hidden_neurons = 20
in_out_neurons = 1
epochs = 100
batch_size = 3


# In[ ]:

def FCNN(hidden_neurons=20):
    model = Sequential()
    model.add(Dense(hidden_neurons,input_dim=12))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=1))
    model.add(Activation("linear"))
    return model


# In[ ]:

model = FCNN()
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])
es = EarlyStopping(monitor='val_loss', patience=2)
csv_logger = CSVLogger('training.log')          
hist = model.fit(trainX,trainY,nb_epoch=epochs, batch_size=batch_size,
                    verbose=1,validation_data=(testX, testY),callbacks=[es, csv_logger],shuffle=False)
scores = model.evaluate(testX, testY, batch_size=batch_size,verbose=0)

print('test score:', scores[0])
print('test accuracy:', scores[1])


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#Plot
plt.legend(loc='best')
plt.grid()
plt.xlabel("predict")
plt.ylabel("actual")
plt.plot(trainPredict,trainY)
plt.show()
plt.savefig("train_fcnn.png")

plt.legend(loc='best')
plt.grid()
plt.xlabel("predict")
plt.ylabel("actual")
plt.plot(testPredict,testY)
plt.show()
plt.savefig("test_fcnn.png")

loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='acc')
plt.plot(range(epochs), val_loss, marker='.', label='val_acc')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
plt.savefig("acc_fcnn.png")

#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))


# In[ ]:

def Simple_LSTM(hidden_neurons=20):
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_dim=1))
    model.add(Dense(1))
    model.add(Activation("linear"))
    return model



model = Simple_LSTM()
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])
es = EarlyStopping(monitor='val_loss', patience=2)
csv_logger = CSVLogger('training.log')          
hist = model.fit(trainX,trainY,nb_epoch=epochs, batch_size=batch_size,
                    verbose=1,validation_data=(testX, testY),callbacks=[es, csv_logger],shuffle=False)
scores = model.evaluate(testX, testY, batch_size=batch_size,verbose=0)

print('test score:', scores[0])
print('test accuracy:', scores[1])


# In[ ]:

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


plt.legend(loc='best')
plt.grid()
plt.xlabel("predict")
plt.ylabel("actual")
plt.plot(trainPredict,trainY)
plt.show()
plt.savefig("train_slstm.png")

plt.legend(loc='best')
plt.grid()
plt.xlabel("predict")
plt.ylabel("actual")
plt.plot(testPredict,testY)
plt.show()
plt.savefig("test_slstm.png")

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='acc')
plt.plot(range(epochs), val_loss, marker='.', label='val_acc')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
plt.savefig("acc_slstm.png")


# In[ ]:

def Dropout_LSTM(hidden_neurons=20):
    model = Sequential()
    model.add(LSTM(hidden_neurons, input_dim=12))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("linear"))
    return model

model = Dropout_LSTM()
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])
es = EarlyStopping(monitor='val_loss', patience=2)
csv_logger = CSVLogger('training.log')          
hist = model.fit(trainX,trainY,nb_epoch=epochs, batch_size=batch_size,
                    verbose=1,validation_data=(testX, testY),callbacks=[es, csv_logger],shuffle=False)
scores = model.evaluate(testX, testY, batch_size=batch_size,verbose=0)

print('test score:', scores[0])
print('test accuracy:', scores[1])


# In[ ]:

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
              
              
plt.legend(loc='best')
plt.grid()
plt.xlabel("predict")
plt.ylabel("actual")
plt.plot(trainPredict,trainY)
plt.show()
plt.savefig("train_dlstm.png")

plt.legend(loc='best')
plt.grid()
plt.xlabel("predict")
plt.ylabel("actual")
plt.plot(testPredict,testY)
plt.show()
plt.savefig("test_dlstm.png")

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='acc')
plt.plot(range(epochs), val_loss, marker='.', label='val_acc')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
plt.savefig("acc_dlstm.png")


# In[ ]:

def B_LSTM(hidden_neurons=20):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_neurons,input_dim=12)))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("linear"))
    return model

model = B_LSTM()
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["accuracy"])
es = EarlyStopping(monitor='val_loss', patience=2)
csv_logger = CSVLogger('training.log')          
hist = model.fit(trainX,trainY,nb_epoch=epochs, batch_size=batch_size,
                    verbose=1,validation_data=(testX, testY),callbacks=[es, csv_logger],shuffle=False)
scores = model.evaluate(testX, testY, batch_size=batch_size,verbose=0)

print('test score:', scores[0])
print('test accuracy:', scores[1])


# In[ ]:

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
              
              
plt.legend(loc='best')
plt.grid()
plt.xlabel("predict")
plt.ylabel("actual")
plt.plot(trainPredict,trainY)
plt.show()
plt.savefig("train_blstm.png")

plt.legend(loc='best')
plt.grid()
plt.xlabel("predict")
plt.ylabel("actual")
plt.plot(testPredict,testY)
plt.show()
plt.savefig("test_blstm.png")

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='acc')
plt.plot(range(epochs), val_loss, marker='.', label='val_acc')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()
plt.savefig("acc_blstm.png")

