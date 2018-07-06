import numpy as np

def K_folded_data(X, y, k):
	'''
	This is the function that would take in X, y and randomly split them into number of folds
	as specified for cross-validation
	
	Inputs:
	X: Training data
	y: Labels of the training data
	k: Number of folds the user want to have
	outputs:
	X_folds: Split training data 
	y_folds: Split label accordingly
	'''
	num_training = X.shape[0]
	X_folds = []
	y_folds = []
	
	indexRandom = np.arange(num_training)
	np.random.shuffle(indexRandom)

	X_random = X[indexRandom[:]]
	y_random = y[indexRandom[:]]
	X_folds, y_folds = np.array_split(X_random, k), np.array_split(y_random.reshape(-1, 1), k)
	
	return X_folds, y_folds