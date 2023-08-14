import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import *

"""
The KernelRidgeRegression class implements the Kernel Ridge Regression closed form solution. It is a non-parametric 
learning algorithm that produces a predictor f^*(.) which is a function in a Reproducing Kernel Hilbert Space (RKHS). 
The representer theorem states that even if we’re trying to solve an optimization problem in an RKHS containing linear
combinations of functions K(x_i, .) centered in n arbitrary points x_i, if we take x_t in X (design matrix) as those points,
then the solution lies in the m dimensional space generated by the columns K(x_t, .) of the kernel matrix K (t=1,...,m).
Furthermore, predictions f^*(x) can be expressed only in terms of inner products, therefore we do not have the need 
for an explicit K(x_t, .), we need only K(x_t1, x_t2) with x_t1,x_t2 in X.
"""

class KernelRidgeRegression(BaseEstimator, RegressorMixin):

    def __init__(self, alpha):
        """
        Initializes the KernelRidgeRegression object.
        
        Parameters:
        - alpha: tuple containing the regularization factor and the kernel bandwidth.
        """
        self.kernel = 'rbf'             # Kernel type (default is Radial Basis Function, i.e. gaussian kernel)
        self.alpha = alpha[0]           # Regularization factor
        self.gamma = alpha[1]           # Bandwidth of the kernel
        self.X = None                   # Training design matrix
        self.y = None                   # Training label vector
        self.w = []                     # Weights/coefficients of the optimal model
        self.verbose = True             # Flag to determine if verbose outputs should be printed.


    def fit(self, X, y):
        """
        Trains the KRR predictor.
        
        The method initializes the design matrix and the target vector. It computes the kernel matrix 
        and subsequently determines the optimal predictor's parameters utilizing the PLU factorization.

        Parameters:
        - X: Training design matrix
        - y: TRaining labels vector
        """
        self.X = X.to_numpy()
        self.y = y.to_numpy() 
        
        if self.verbose :
            print("\n-------------------------------------Fitting Kernel Ridge Regression-------------------------")

        K = self.kernel_matrix(type='full')

        if self.verbose :
            print("\n-------------------------------------Kernel matrix has been created--------------------------")

        ID = np.identity(self.X.shape[0])
        # Decomposizione LU
        P, L, U = sp.linalg.lu(self.alpha*ID + K)
        # Calcolo dell'inversa utilizzando la decomposizione LU 
        self.w = np.dot(y.T, sp.linalg.solve(U, sp.linalg.solve(L, P))).reshape(-1)
        #self.w = np.dot(self.y.T, np.linalg.inv(U) @ np.linalg.inv(L) @ P).reshape(-1) 

        if self.verbose :
            print("\n-------------------------------------Inverse matrix has been computed------------------------")


    def predict(self, Xp):
        """
        Makes predictions using the trained Kernel Ridge Regression model.
        
        Parameters:
        - Xp: Training/test design matrix
        
        Returns:
        - Predicted values
        """
        Kx = self.kernel_matrix(Xp, type='full')
        return np.dot(self.w, Kx)
    

    def score(self, X, y, method=mean_squared_error):
        """
        Evaluates the model performance using a given method (default is mean squared error).
        
        Parameters:
        - X: Training/test design matrix
        - y: Training/test true labels
        - method: evaluation metric
        
        Returns:
        - Predictor performance score
        """
        X = X.to_numpy()
        y = y.to_numpy()
        y_hat = self.predict(X)
        return method(y, y_hat)
    

    def kernel_matrix(self, Xp=None, type='full'):
        """
        Computes the kernel matrix for the given data.
        
        Parameters:
        - Xp: Training/test design matrix (if it is None, it will be set equal to self.X)
        - type: Type of matrix ('full' to compute K(x_i,x_j), with x_i,x_j in self.X, 'redux' to compute K(xp_i,xp_j),
                with x_i,x_j in Xp)
        
        Returns:
        - Kernel matrix
        """
        X = self.X if type=='full' else Xp
        Xp = X if Xp is None else Xp
        if self.kernel == 'rbf' or self.kernel == 'gaussian':
            return np.exp(-1/(2*self.gamma) * np.sum((X[:, np.newaxis] - Xp) ** 2, axis=-1)) #* cdist(self.X, Xp, metric='euclidean') 
            #return rbf_kernel(self.X, Xp, gamma=1.0/(2*self.gamma))
        else:
            raise ValueError("Unsupported kernel: "+self.kernel)

       
    def set_X(self, X):
        self.X = X

    def set_kernel(self, kernel):
        self.kernel = kernel

    def set_alpha(self, alpha):
        self.alpha = alpha
    
    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_w(self, w):
        self.w = w
    
    def set_verbose(self, verbose):
        self.verbose = verbose
        
    def get_kernel(self):
        return self.kernel
    
    def get_alpha(self):
        return self.alpha
    
    def get_gamma(self):
        return self.gamma
    
    def get_w(self):
        return self.w

