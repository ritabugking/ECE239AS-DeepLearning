import numpy as np
import h5py

def data_loader(datapath, num_of_file):
	'''
	This is a function that will take inputs from user and load the assigned number
	of dataset to the program in Tensors;
	inputs:
		1. Directory of the dataset in the format of 'dir/dir/.../dir';
		2. Number of data set we want to use with a minimum of 1, maximum of 9;
	output:
		X: with dimension of (288*num_of_file, 25, 313);
		y: with dimension of (288*num_of_file,);
	'''
	X, y = None, None
	for i in range(1, num_of_file + 1):
		curpath = datapath + '/A0' + str(i) + 'T_slice.mat'
		curFile = h5py.File(curpath, 'r')
		curX = np.copy(curFile['image'])
		cury = np.copy(curFile['type'])
		cury = cury[0, 0:curX.shape[0]:1]
		cury = np.asarray(cury, dtype=np.int32)
		if i == 1:
			X = curX
			y = cury
		else:
			X = np.concatenate([X, curX])
			y = np.concatenate([np.asarray(y), np.asarray(cury)])
	# y = y[0, 0:X.shape[0]:1]
	# y = np.asarray(y, dtype=np.int32)		
		
	return X, y