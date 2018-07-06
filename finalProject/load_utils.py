from __future__ import print_function
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split

def remove_nan(X, y):
    M,N,K = X.shape
    X = X.reshape(M, -1)
    y = y[~np.isnan(X).any(axis=1)]
    temp = X[~np.isnan(X).any(axis=1)]
    X = temp.reshape(temp.shape[0], N, K)
    return X,y

def load_one(fname):
    """
    Reads and returns a vector from a file.
    
    Parameters
    --------------------
        fname  -- string, filename
        
    Returns
    --------------------
        X -- (288,25,1000)
        y -- (288,)
    """
    A01T = h5py.File(fname, 'r')
    X = np.copy(A01T['image'])
    y = np.copy(A01T['type'])
    y = y[0,0:X.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    return X, y

def load_one_file(file_num):
	file_name = 'A0' + str(file_num) + 'T_slice.mat'
	cur_data = h5py.File(file_name, 'r')
	X = np.copy(cur_data['image'])[:, 0:22, :]
	y = np.copy(cur_data['type'])[0,0:X.shape[0]:1]
	y = np.asarray(y)
	print('Data loaded from' + file_name + ':')
	print(X.shape)
	print(y.shape)
	
	return X,y

def load_all(path):
    files = os.listdir(path)
    list_X = []
    list_y = []
    for i in range(9):
        fname = path + 'A0' + str(i+1) +'T_slice.mat'
        X,y = load_one(fname)
        X,y = remove_nan(X,y)
        list_X.append(X)
        list_y.append(y)
    return list_X, list_y

#split data into testdata(#test_size) + rest(train + val)
def split_test(test_size, X, y):
	X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size = test_size)

	return X_rest, X_test, y_rest, y_test

#split data into val data + train data
def split_val(val_size, X, y):
	X_rest, X_val, y_rest, y_val = train_test_split(X, y, test_size = val_size)

	return X_rest, X_val, y_rest, y_val


def self_train_val_test(file_num, test_size, val_size):
		X_ori, y_ori = load_one_file(file_num)
		X, y = remove_nan(X_ori, y_ori)
		X_rest1, X_test, y_rest1, y_test = split_test(test_size, X, y)
		X_rest2, X_val, y_rest2, y_val = split_val(val_size, X_rest1, y_rest1)

		X_train = X_rest2
		y_train = y_train2

		return X_train, X_val, X_test, y_train, y_val, y_test

#Draw trainng/validation data from whole set, draw testing data from specific subject data set
def rd_train_val_self_test(path, fnum):
    list_X, list_y = load_all(path)
    X = list_X[fnum-1]
    y = list_y[fnum-1]
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=50, random_state=42)
    for i in range(len(list_X)):
        if i != fnum-1:
            X_rest = np.concatenate((X_rest, list_X[i]), axis=0)
            y_rest = np.concatenate((y_rest, list_y[i]), axis=0)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=100, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test 

