import numpy as np
import pandas as pd

"""
    TargetEncoding class applies target encoding to categorical variables. The target encoding involves replacing 
    each value x_i of the i-th category with the average/median of the corresponding target values y_j(x_i) (j = 1, ..., |y_(x_i)|) 
    in the training set. For the test set, the values are replaced with the averages calculated from the training set.
    The encoding can include an optional smoothing.
"""

class TargetEncoding:

    def __init__(self, target_col, encoding_type, smoothing=0, default_shrink=1/6):
        """
        Initializes the TargetEncoding object.

        """
        self.target_col = target_col            #Name of target column.
        self.encoding_type = encoding_type      #Encoding type to use ('mean' or 'median').
        self.smoothing = smoothing              #Smoothing factor for computing the encoding values.
        self.default_shrink = default_shrink    #The shrinkage factor to determine default values for missing categories during transformation
        self.encoding_params = []               #Encoding parameters for each categorical column
        self.defaults = []                      #Default values for each categorical column
    

    def fit_transform(self, data, encoding_cols):
        """
        Fits the encoder using the provided data (training set) and returns the transformed dataset.
        
        Parameters:
        - data: Input DataFrame.
        - encoding_cols: List of columns to be encoded.
        
        Returns:
        - DataFrame with encoded columns.
        """
        self.encoding_params = []
        for col in encoding_cols:
            center_index = self.compute_center_index(data, col)
            self.encoding_params.append(center_index)
            self.defaults.append(self.default_shrink*np.percentile(center_index.to_numpy(), 10))
        for i, col in enumerate(encoding_cols):
            data[col] = data[col].map(self.encoding_params[i])
            if (data[col] == 0).sum() > 0:
                data[col].replace(to_replace = 0, value = 0.001, inplace=True)
        return data[encoding_cols]
    

    def transform(self, data, encoding_cols):
        """
        Transforms the input data (test set/validation set) using the pre-computed encoding parameters.
        
        Parameters:
        - data: Input DataFrame.
        - encoding_cols: List of columns to be encoded.
        
        Returns:
        - DataFrame with encoded columns.
        """
        for i, col in enumerate(encoding_cols):
            data[col] = data[col].map(self.encoding_params[i]).fillna(self.defaults[i])
            if (data[col] == 0).sum() > 0:
                data[col].replace(to_replace = 0, value = 0.001, inplace=True)
        return data[encoding_cols]
    
    
    def compute_center_index(self, data, col):
        """
        Computes the 'center index' for each vector y_(x_i), based on the specified encoding type.
        
        Parameters:
        - data: Input DataFrame.
        - col: Column for which to compute the center index.
        
        Returns:
        - Center index for the given column.
        """
        data_grouped = data.groupby(col)[self.target_col]
        prior_mean = data[self.target_col].mean().squeeze()
        prior_count = data_grouped.count().squeeze()
        if self.encoding_type == 'mean' :
            center_index_est = data_grouped.mean().squeeze()
        elif self.encoding_type == 'median':
            center_index_est = data_grouped.median().squeeze()
        else:
            raise ValueError("Invalid encoding_type. Supported values: 'median', 'mean'")
        return (prior_count * center_index_est + self.smoothing * prior_mean) / (prior_count + self.smoothing)
    
    
    def get_encoding_params(self):
        return self.encoding_params
    
    def get_encoding_type(self):
        return self.encoding_type
    
    def get_target_col(self):
        return self.target_col

