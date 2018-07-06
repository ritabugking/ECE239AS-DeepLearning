from __future__ import print_function
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import Dense, LSTM
from keras import optimizers
import h5py
import data_loader as dl
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, LSTM, Reshape
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

classifier = Sequential()

# Step 1 - Convolution

classifier.add(Conv2D(25, (1, 10), input_shape = (22, 1000, 1), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(25, (1, 10),  activation = 'relu'))
classifier.add(BatchNormalization())
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (1, 4)))
classifier.add(Dropout(0.25))

# Adding a second convolutional layer

classifier.add(Conv2D(50, (1, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(50, (1, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (1, 2)))
classifier.add(Dropout(0.25))

# Step 3 - Flattening

classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.25))
#classifier.add(Dense(units = 4, activation = 'softmax'))
classifier.add(Reshape(1,128))
#classifier.add(LSTM(128, input_shape=(1,128), output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
classifier.add(LSTM(128, input_shape=(1, 128), activation="sigmoid", return_sequences=True, units=256, recurrent_activation="hard_sigmoid"))
#classifier.add(LSTM(128, input_shape=(1, 128), activation="sigmoid", return_sequences=True, recurrent_activation="hard_sigmoid"))

classifier.add(Dropout(0.5))
classifier.add(LSTM(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
classifier.add(Dropout(0.5))
classifier.add(Dense(4, activation='softmax'))

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
classifier.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

# Compiling the CNN
#adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#classifier.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

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





#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 50)
X_train = X_train.reshape(-1,22,1000,1)
X_test = X_test.reshape(-1,22,1000,1)

classifier.fit(X_train, y_train,
          batch_size=24,
          epochs=5,
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