import numpy
import pandas
from matplotlib import pyplot

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import random

data = pandas.read_excel("./data/data1.xlsx", sheetname='Sheet1').as_matrix()

X = data[1:,0]
Y = data[1:,1]

#x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

variables = data[1:,:]

cur_act = activation='linear'

model = Sequential()
#model.add(Dense(20, input_shape=(2,), init='uniform', activation = cur_act))
model.add(Dense(20, input_dim = 1, activation = cur_act))
model.add(Dense(10, init='uniform', activation = cur_act))
model.add(Dense(1, init='uniform', activation = cur_act))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath="./data/best_model.hdf5", verbose=0, save_best_only=True) #save best model



#tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)


batch_size = 100
epochs = 100

#model.fit(X, Y, batch_size = batch_size, epochs=epochs, validation_split=0.25, verbose=2, shuffle=True, callbacks=[tbCallBack])
#model.fit(X, Y, batch_size = batch_size, epochs=epochs, validation_split=0.25, verbose=2, shuffle=True)
model.fit(X, Y, batch_size = batch_size, epochs=epochs, validation_split=0.25, verbose=2, shuffle=True, callbacks=[monitor, checkpointer])
model.load_weights('./data/best_model.hdf5')

pred = model.predict(x_test)

print pred

pyplot.plot(x_test,pred,color='red')
pyplot.draw()

pyplot.show()


#model.save('./data/my_first_model.h5')
#del model

#pyplot.plot(X,Y)
#pyplot.show()
