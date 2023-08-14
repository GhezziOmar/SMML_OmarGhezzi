import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import *

"""
The RidgeRegression class implements Ridge Regression closed form solution. It inherits from 
BaseEstimator and RegressorMixin classes (allowing model comparison) and provides 
methods for training the model, making predictions and evaluating the model (besides the 
setter and getter methods).
"""

class RidgeRegression(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1.0):
        """
        Initializes the RidgeRegression object.
        
        Parameters:
        - alpha: Regularization factor.
        """
        self.alpha = alpha
        self.w = []

     
    def fit(self, X, y):
        """
        Fits/trains the Ridge Regression model using the given training data.
        
        Parameters:
        - X: Training design matrix
        - y: Training labels vector
        """
        X = X.to_numpy()
        y = y.to_numpy()        
        X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=-1)
        ID = np.identity(X.shape[1])
        self.w = np.dot(np.dot(np.linalg.inv(self.alpha*ID + np.dot(X.T, X)), X.T), y).reshape(-1)        
    
    
    def predict(self, X):
        """
        Makes predictions using the trained Ridge Regression model.
        
        Parameters:
        - X: Trainng/test design matrix
        
        Returns:
        - Predicted values
        """
        X = X.to_numpy()       
        X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=-1)
        return np.dot(X, self.w)
    
    
    def score(self, X, y, method=mean_squared_error):
        """
        Evaluates the model performance using a given method (default is mean squared error).
        
        Parameters:
        - X: Trainng/test design matrix
        - y: True labels
        - method: Evaluation metric
        
        Returns:
        - Predictor performance score
        """
        y = y.to_numpy()
        y_hat = self.predict(X)
        return method(y, y_hat)
    
    
    def set_w(self, w):
        self.w = w
    
    def set_alpha(self, alpha):
        self.alpha = alpha
        
    def get_w(self):
        return self.w
    
    def get_alpha(self):
        return self.alpha
    
