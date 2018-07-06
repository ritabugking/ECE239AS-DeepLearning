from __future__ import print_function

import numpy as np
import h5py
import data_loader as dl
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import math


#def load all the data
def load_all_data():
	A01T = h5py.File('A01T_slice.mat', 'r')
	X = np.copy(A01T['image'])
	X = X[:, 0:22, :]
	y = np.copy(A01T['type'])
	y = y[0,0:X.shape[0]:1]
	#y = np.asarray(y)

	for i in range(2,10):
		file_name = 'A0' + str(i) + 'T_slice.mat'
		cur_data = h5py.File(file_name, 'r')
		X_temp = np.copy(cur_data['image'])[:,0:22,:]
		X = np.concatenate((X,X_temp), axis=0)
		y_temp = np.copy(cur_data['type'])[0,0:X_temp.shape[0]:1]
		y = np.concatenate((y, y_temp), axis=0)
		y = np.asarray(y)
	print(X.shape)
	print(y.shape)

	return X, y

def load_data_from(file_num):
	file_name = 'A0' + str(file_num) + 'T_slice.mat'
	cur_data = h5py.File(file_name, 'r')
	X = np.copy(cur_data['image'])[:, 0:22, :]
	y = np.copy(cur_data['type'])[0,0:X.shape[0]:1]
	y = np.asarray(y)
	print(X.shape)
	print(y.shape)

	return X,y

#Load data
# A01T = h5py.File('A01T_slice.mat', 'r')
# X = np.copy(A01T['image'])
# X = X[:, 0:22, :]
# y = np.copy(A01T['type'])
# y = y[0,0:X.shape[0]:1]
# y = np.asarray(y)
#
# print ("Data shape:")
# print (X.shape)
# print (y.shape)

X, y = load_data_from(1)
y = LabelEncoder().fit_transform(y)
y = OneHotEncoder(sparse = False).fit_transform( y.reshape(-1,1) )

def get_on_hot(number):
	on_hot = [0] * 4
	on_hot[int(number-769)] = 1
	return on_hot

#Change the result representation
y=np.array(map(get_on_hot, y))

#Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 50)



#Training parameters
training_steps = 12
batch_size = 48
display_step = 4

#rnn network parameters
num_input = 22 #313 sequences of 22 electrodes' data
timesteps= 1000
num_hidden = 50
num_classes = 4 #Four number to represent result

#tf graph in/output
X_in = tf.placeholder(tf.float64, [None, timesteps, num_input])
Y_out = tf.placeholder(tf.float64, [None, num_classes])


#define weights
weights = {
	'out': tf.Variable(tf.random_normal([num_hidden, num_classes],dtype=tf.float64), dtype=tf.float64)
}

biases = {
	'out': tf.Variable(tf.random_normal([num_classes],dtype=tf.float64),dtype=tf.float64)
}

def RNN(x, weights, biases):
	# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, timesteps, 1)

	#Define a lstm cell with tensorflow
	lsmt_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

	#Get cell states + cell output
	outputs, states = rnn.static_rnn(lsmt_cell, x, dtype=tf.float64)

	#linear activtion
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X_in, weights, biases)
prediction = tf.nn.softmax(logits)

#Loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_out))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

#Evaluation
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_out,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))


#Initialization of tf variables
init = tf.global_variables_initializer()
#Start Training
with tf.Session() as sess:
	sess.run(init)

	for step in range(1, training_steps+1):
		X_batch = None #[batch_size, timestep, num_input]
		Y_batch = None #[batch_size]

		#Draw batch from original imported data
		ind = np.random.choice(238, batch_size)
		X_batch = X_train[ind,:,:]
		Y_batch = y_train[ind]
		#resize the data
		X_batch = np.swapaxes(X_batch, 1, 2)
		#Optimization
		sess.run(train_op, feed_dict={X_in:X_batch, Y_out:Y_batch})

		if step % display_step == 0 or step == 1:
			loss, acc = sess.run([loss_op, accuracy], feed_dict={X_in: X_batch,
                                                                 Y_out: Y_batch})
			print("Step " + str(step) + ", Minibatch Loss= " + \
				  "{:.4f}".format(loss) + ", Training Accuracy= " + \
				  "{:.3f}".format(acc))
	print("Optimization Done")


