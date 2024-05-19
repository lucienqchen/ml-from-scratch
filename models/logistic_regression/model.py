import numpy as np
from scripts.util import symbol_name

from scripts.util import *

class Logistic_Regression():
    
    def __init__(self):
        
        """
            Some initializations, if neccesary
        """
        
        self.model_name = 'Logistic Regression'
    
    def fit(self, X_train, y_train):
        
        """
            Save the datasets in our model, and do normalization to y_train
            
            Parameter:
                X_train: Matrix or 2-D array. Input feature matrix.
                Y_train: Matrix or 2-D array. Input target value.
        """
        
        self.X = X_train
        self.y = y_train
        
        count = 0
        uni = np.unique(y_train)
        for y in y_train:
            if y == min(uni):
                self.y[count] = -1
            else:
                self.y[count] = 1
            count += 1        
        
        n,m = X_train.shape
        self.theta = np.zeros(m)
        self.b = 0
    
    def gradient(self, X_inp, y_inp, theta, b):
        
        """
            Calculate the gradient of Weight and Bias, given sigmoid_yhat, true label, and data

            Parameter:
                X_inp: Matrix or 2-D array. Input feature matrix.
                y_inp: Matrix or 2-D array. Input target value.
                theta: Matrix or 1-D array. Weight matrix.
                b: int. Bias.

            Return:
                grad_theta: gradient with respect to theta
                grad_b: gradient with respect to b
        """
        
        grad_b = 0
        grad_theta = np.zeros_like(self.theta)
        y_hat = sigmoid(np.dot(X_inp, theta) + b)
        diff = y_hat - y_inp
        grad_b = np.mean(diff)
        grad_theta = np.dot(X_inp.T, diff)
        
        return grad_theta, grad_b

    def gradient_descent_logistic(self, alpha, num_pass, early_stop=0, standardized = True):
        
        """
            Logistic Regression with gradient descent method

            Parameter:
                alpha: (Hyper Parameter) Learning rate.
                num_pass: Number of iteration
                early_stop: (Hyper Parameter) Least improvement error allowed before stop. 
                            If improvement is less than the given value, then terminate the function and store the coefficents.
                            default = 0.
                standardized: bool, determine if we standardize the feature matrix.
                
            Return:
                self.theta: theta after training
                self.b: b after training
        """
        
        if standardized:
            self.X = z_standardize(self.X)
        
        n, m = self.X.shape

        self.loss = []
        
        for i in range(num_pass):    
            
            grad_theta, grad_b = self.gradient(self.X, self.y, self.theta, self.b)
            temp_theta = self.theta - (alpha * grad_theta)
            temp_b = self.b - (alpha * grad_b)
            previous_y_hat = sigmoid(np.dot(self.X, self.theta) + self.b)
            temp_y_hat = sigmoid(np.dot(self.X, temp_theta) + temp_b)
            
            pre_error = logistic_loss(self.y, previous_y_hat)
            temp_error = logistic_loss(self.y, temp_y_hat)
            
            if (abs(pre_error - temp_error) < early_stop) | (abs(abs(pre_error - temp_error) / pre_error) < early_stop):
                return temp_theta, temp_b
            
            if temp_error <= pre_error:
                self.theta = temp_theta
                self.b = temp_b
                alpha *= 1.3
            else:
                alpha *= 0.9
                
            self.loss.append(pre_error)
            
        return self.theta, self.b
    
    def predict_ind(self, x: list):
        
        """
            Predict the most likely class label of one test instance based on its feature vector x.

            Parameter:
            x: Matrix, array or list. Input feature point.
            
            Return:
                p: prediction of given data point
        """
        p = sigmoid(np.dot(x, self.theta) + self.b)                     # -------- calculate probability (you can use the sigmoid function)
        
        return p
    
    def predict(self, X):
        
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Parameter:
            x: Matrix, array or list. Input feature point.
            
            Return:
                p: prediction of given data matrix
        """
          
        ret = [self.predict_ind(x) for x in X]                  # -------- Use predict_ind to generate the prediction list
        
        return ret
    