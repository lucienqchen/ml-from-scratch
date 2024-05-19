import numpy as np

class LinearRegression():
    
    def __init__(self, alpha=1e-10, num_iter=10000, early_stop=1e-50, intercept=True, init_weight=None):
        
        """
        Initialize the Linear Regression model.

        Parameters:
            alpha (float): The learning rate for gradient descent. Default is 1e-10.
            num_iter (int): The number of iterations to update the coefficients with training data. Default is 10000.
            early_stop (float): A constant used to control early stopping. Default is 1e-50.
            intercept (bool): Whether to fit an intercept term. Default is True.
            init_weight (ndarray): An optional matrix of shape (n x 1) to initialize the weights for testing purposes.

        Attributes:
            model_name (str): The name of the model, which is 'Linear Regression'.
            alpha (float): The learning rate.
            num_iter (int): The number of iterations.
            early_stop (float): The early stopping constant.
            intercept (bool): Whether an intercept term is fitted.
            init_weight (ndarray): The initial weights for testing purposes.
        """
        
        self.model_name = 'Linear Regression'
        
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.intercept = intercept
        self.init_weight = init_weight  ### For testing correctness.
        
    
    def fit(self, X_train, y_train):
        
        """
        Fits the linear regression model to the training data.

        Parameters:
        - X_train (numpy.ndarray): The input features of the training data.
        - y_train (numpy.ndarray): The target values of the training data.

        Returns:
        - None

        This method updates the internal state of the linear regression model by setting the input features, target values,
        and the number of training samples. If the `intercept` flag is set to True, it adds a column of ones to the input
        features. It then initializes the coefficients either with the provided initial weights or with random values
        between -1 and 1. Finally, it calls the `gradient_descent` method to optimize the coefficients.
        """
           
        self.X = X_train
        self.y = y_train
        self.n = len(y_train)
        
        if self.intercept:
            ones = np.ones((self.n, 1)) # initializes an n x 1 column of ones
            self.X = np.hstack([ones, self.X]) # adds the column of ones to the first column
        
        if isinstance(self.init_weight, np.ndarray):
            self.coef = np.array(self.init_weight).reshape(self.X.shape[1],)
        else:
            self.coef = np.random.uniform(-1, 1, size=self.X.shape[1]) #### Please change this after you get the example right.
        self.gradient_descent()
        
    def gradient(self):
        """
            Helper function to calculate the gradient respect to coefficient.
        """
        preds = np.dot(self.X, self.coef) # calculates wX
        errors = self.y - preds # e = Y - (wX + b)
        self.grad_coef = -2 * np.dot(self.X.T, errors) # -2X * (Y - (wX + b))

    def gradient_descent(self):
        
        """
        Performs gradient descent optimization to update the coefficients of the linear regression model.
        
        Returns:
            self: The updated linear regression model object.
        """
        
        self.loss = []
        
        for i in range(self.num_iter): 
                
            self.gradient()
            
            previous_y_hat = np.dot(self.X, self.coef) # Y_hat = wX + b
            
            temp_coef = self.coef - self.alpha * self.grad_coef # provides updated coefficients
            
            current_y_hat = np.dot(self.X, temp_coef) # new Y_hat = new_wX + new_b
            
            ones = np.ones((1, self.n))  # Matrix with 1's (1 x n), help with calculate the sum of a mattrix. hint: Think about dot product.
            
            pre_error = np.sum((self.y - previous_y_hat) ** 2) # sum(Y - (wX + b) ** 2)
            
            current_error = np.sum((self.y - current_y_hat) ** 2) # sum(Y - (new_wX + new_b) ** 2)
            
            ### This is the early stop, don't modify fllowing three lines.
            if (abs(pre_error - current_error) < self.early_stop) | (abs(abs(pre_error - current_error) / pre_error) < self.early_stop):
                self.coef = temp_coef
                return self
            
            if current_error <= pre_error:
                self.coef = temp_coef
                self.alpha = self.alpha * 1.3 # if the loss improved, increase alpha
            else:
                self.alpha = self.alpha * 0.9 # if loss did not improve, decrease alpha
            
            self.loss.append(current_error)
            
            if i % 10000 == 0:
                print('Iteration: ' +  str(i))
                print('Coef: '+ str(self.coef))
                print('Loss: ' + str(current_error))
                       
        return self
    
    def ind_predict(self, x: list):
        """
            Predict the value based on its feature vector x.

            Parameter:
            x: Matrix, array or list. Input feature point.
            
            Return:
                result: prediction of given data point
        """
        result = np.matmul(self.coef, x) # yi = w*xi + b
        
        return result
    
    def predict(self, X):
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Parameter:
            X: Matrix, array or list. Input feature point.
            
            Return:
                ret: prediction of given data matrix
        """
        
        ret = []
        if self.intercept:
            ones = np.ones((self.n, 1)) 
            X = np.hstack([ones, X])
        for x in X:
            ret.append(self.ind_predict(x))
        return ret
        