import numpy as np
import pandas as pd
from KernelRidgeRegression import KernelRidgeRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import *

"""
The DCKRR (Divide and Conquer Kernel Ridge Regression) class offers an implementation of the kernel ridge regression 
with a divide-and-conquer strategy. This strategy allows handling large datasets by partitioning them into smaller 
subsets, training separate KRR models on each, and then combining their results.
The divide-and-conquer approach employed in DCKRR tries to handle large datasets while maintaining 
the advantages of the kernel ridge regression's non-parametric approach.
"""

class DCKRR(BaseEstimator, RegressorMixin):

    def __init__(self, alpha):
        """
        Initializes the DCKRR object.
        
        Parameters:
        - alpha: A tuple containing regularization strength and the kernel bandwidth.
        """
        self.alpha = alpha                          #Regularization parameter.
        self.num_subsets = 4                        #The number of subsets to divide the dataset into.
        self.models = []                            #Stores the trained KRR models for each subset.
        self.ws = []                                #Contains the weights of the predictors for each subset.
        self.verbose = False                        # Flag to determine if verbose outputs should be printed.


    def fit(self, X, y):
        """
        Divides the dataset into subsets and trains a KRR model on each.

        Parameters:
        - X: Full training design matrix
        - y: Full taining labels vector
        """

        X_subs = np.array_split(X, self.num_subsets)
        y_subs = np.array_split(y, self.num_subsets)

        for i in range(self.num_subsets):
            X_sub = X_subs[i]
            y_sub = y_subs[i]

            if self.verbose :
                print("\n=======================Kernel Ridge Regression (subset n. "+str(i)+")=====================")
            
            model = KernelRidgeRegression(self.alpha)

            self.models.append(model)
            self.models[i].set_verbose(self.verbose)
            self.models[i].fit(X_sub, y_sub)
            self.ws.append(self.models[i].get_w())

    
    def predict_single_example(self, X):
        """
        Predicts the target for a single input instance using the trained models.
        
        Parameters:
        - Xp: Training/test design matrix
        
        Returns:
        - Predicted values
        """
        y_hat = np.zeros([self.num_subsets, X.shape[0]])
        for i, x in enumerate(X):

            if self.verbose :
                    print("\n=============================Score for Prediction n. "+str(i)+"==========================")

            for j, model in enumerate(self.models):

                if self.verbose :
                    print("\n--------------------------Score for submodel n. "+str(j)+"------------------------")
                y_hat[j,i] = np.dot(self.ws[j], model.kernel_matrix(x, type='full'))

        return np.mean(y_hat, 0)
    
    
    def score(self, X, y, method=mean_squared_error):
        """
        Evaluates the combined performance of the models using a specified metric.
        
        Parameters:
        - X: Training/test design matrix
        - y: Training/test true labels
        - method: evaluation metric
        
        Returns:
        - Predictor performance score
        """
        X = X.to_numpy()
        y = y.to_numpy()
        y_hat = self.predict_single_example(X)
        return method(y, y_hat)  
    

    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def set_ws(self, ws):
        self.ws = ws

    def set_num_subsets(self, num_subsets):
        self.num_subsets = num_subsets
    
    def set_verbose(self, verbose):
        self.verbose = verbose
    
    def get_hyper(self):
        return self.alpha
    
    def get_ws(self):
        return self.ws
    
    def get_w(self):
        return np.mean(self.ws, 0)
