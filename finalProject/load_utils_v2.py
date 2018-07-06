import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
######################################################################
# functions -- input/output
######################################################################

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

def remove_nan(X, y):
    M,N,K = X.shape
    X = X.reshape(M, -1)
    y = y[~np.isnan(X).any(axis=1)]
    temp = X[~np.isnan(X).any(axis=1)]
    X = temp.reshape(temp.shape[0], N, K)
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

# X_train, X_val, X_test, y_train, y_val, y_test  = rd_train_val_self_test('../project_datasets/', 1)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(X_val.shape)
# print(y_val.shape)

