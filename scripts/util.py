import numpy as np

def min_max_normalize(lst):
    
    """
        Helper function for movielens dataset, not useful for discrete multi class clasification.

        Return:
        Normalized list x, in range [0, 1]
    """
    
    maximum = max(lst)
    minimum = min(lst)
    toreturn = []
    
    for i in range(len(lst)):
        toreturn.append((lst[i]- minimum)/ (maximum - minimum))
        
    return toreturn

def z_standardize(X_inp):
    
    """
        Z-score Standardization.
        Standardize the feature matrix, and store the standarize rule.

        Parameter:
        X_inp: Input feature matrix.

        Return:
        Standardized feature matrix.
    """
    
    toreturn = X_inp.copy()
    
    for i in range(X_inp.shape[1]):
        
        std = np.std(X_inp[:, i])               # ------ Find the standard deviation of the feature
        mean = np.mean(X_inp[:, i])             # ------ Find the mean value of the feature
        temp = []
        
        for j in np.array(X_inp[:, i]):
            
            temp += [(j-mean)/std]
        toreturn[:, i] = temp
        
    return toreturn

def sigmoid(z):
    """ 
        Sigmoid Function

        Return:
        transformed x.
    """

    return 1 / (1 + np.exp(-z))

def logistic_loss(y_true, preds):
    one_loss = y_true * np.log(preds + 1e-9)
    zero_loss = (1 - y_true) * np.log(1 - preds + 1e-9)
    return -np.mean(one_loss + zero_loss)
