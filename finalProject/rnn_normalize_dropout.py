from __future__ import print_function
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, LSTM
from keras.layers import Dropout, BatchNormalization
from keras.layers import SimpleRNN, Activation
from keras import optimizers
from sklearn.model_selection import train_test_split
import h5py
import data_loader as dl
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

# Initialising the CNN
model = Sequential()
model2 = Sequential()
model3 = Sequential()
model4 = Sequential()

# Step 1 - Convolution
model.add(BatchNormalization(input_shape=(1,22000)))
model.add(LSTM( output_dim=256, activation='softplus', inner_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(LSTM(output_dim=256, activation='softplus', inner_activation='hard_sigmoid'))
model.add(Dense(4, activation='softmax'))

model2.add(BatchNormalization(input_shape=(1,22000)))
model2.add(LSTM( output_dim=256, activation='softsign', inner_activation='hard_sigmoid', return_sequences=True))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(LSTM(output_dim=256, activation='softsign', inner_activation='hard_sigmoid'))
model2.add(Dense(4, activation='softmax'))

model3.add(BatchNormalization(input_shape=(1,22000)))
model3.add(LSTM( output_dim=256, activation='relu', inner_activation='hard_sigmoid', return_sequences=True))
model3.add(Dropout(0.5))
model3.add(BatchNormalization())
model3.add(LSTM(output_dim=256, activation='elu', inner_activation='hard_sigmoid'))
model3.add(Dense(4, activation='softmax'))

model4.add(BatchNormalization(input_shape=(1,22000)))
model4.add(LSTM( output_dim=256, activation='selu', inner_activation='hard_sigmoid', return_sequences=True))
model4.add(Dropout(0.5))
model4.add(BatchNormalization())
model4.add(LSTM(output_dim=256, activation='selu', inner_activation='hard_sigmoid'))
model4.add(Dense(4, activation='softmax'))

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam2 = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam3 = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam4 = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
model2.compile(loss='categorical_crossentropy', optimizer=adam2, metrics=["accuracy"])
model3.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
model4.compile(loss='categorical_crossentropy', optimizer=adam2, metrics=["accuracy"])

def load_data_from(file_num):
	file_name = 'A0' + str(file_num) + 'T_slice.mat'
	cur_data = h5py.File(file_name, 'r')
	X = np.copy(cur_data['image'])[:, 0:22, :]
	y = np.copy(cur_data['type'])[0,0:X.shape[0]:1]
	y = np.asarray(y)
	# print(X.shape)
	# print(y.shape)

	return X,y

X, y = load_data_from(1)
X = np.nan_to_num(X)
y = LabelEncoder().fit_transform(y)
y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 50)
for i in range(2,10):
    X2, y2 = load_data_from(i)
    X2 = np.nan_to_num(X2)
    y2 = LabelEncoder().fit_transform(y2)
    y2 = OneHotEncoder(sparse = False).fit_transform( y2.reshape(-1,1) )
    X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=50)
    X_train = np.concatenate((X_train,X2_train),axis=0)
    X_test = np.concatenate((X_test, X2_test), axis=0)
    y_train = np.concatenate((y_train, y2_train), axis=0)
    y_test = np.concatenate((y_test, y2_test), axis=0)

X_train = X_train.reshape(-1,1, 22000)
X_test = X_test.reshape(-1,1, 22000)

model.fit(X_train, y_train,
          batch_size=24,
          epochs=2,
          verbose=1,
          validation_data=(X_test, y_test))
model2.fit(X_train, y_train,
          batch_size=24,
          epochs=2,
          verbose=1,
          validation_data=(X_test, y_test))
model3.fit(X_train, y_train,
          batch_size=24,
          epochs=2,
          verbose=1,
          validation_data=(X_test, y_test))
model4.fit(X_train, y_train,
          batch_size=24,
          epochs=2,
          verbose=1,
          validation_data=(X_test, y_test))
# import numpy as np
# #from keras.preprocessing import image
# test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'