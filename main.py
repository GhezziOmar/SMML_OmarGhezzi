import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from sklearn.preprocessing import *
from sklearn.metrics import *
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.linear_model import Ridge
from RidgeRegression import RidgeRegression
from expectedRiskEstimate import expectedRiskEstimate
from KernelRidgeRegression import KernelRidgeRegression
from functions import *
from DCKRR import DCKRR
from TargetEncoding import TargetEncoding
import warnings
warnings.simplefilter("ignore")

def prompt_continue(section_name):
    input(f"Press enter to start {section_name}")

"""
RIDGE REGRESSION AND KERNEL RIDGE REGRESSION ON NUMERICAL FEATURES

OMAR GHEZZI matr. 984352

MAIN: run the data cleaning, pre-processing to the whole dataset. Apply 5-fold
nested cross validation to ridge regression on the resulting pre-processed dataset.
Apply 5-fold best CV estimate to kernel ridge regression (KRR) and DCKRR on
4000 and 83475 examples, respectively.
"""

#==============================================================================
#==                        1)CONTINUOUS FEATURE TRAINING                     ==
#==============================================================================

#================================DATA LOADING=================================
print("\n================================DATA LOADING=================================")

# Load the dataset from the specified CSV file
df = pd.read_csv('./data/dataset.csv')

# Drop the unnecessary columns, namely 'Unnamed: 0' and 'track_id'
df = df.drop(columns=df.columns[0:2])

# Display the first few rows of the dataset to provide an overview
df.head()

# Display statistical information about the dataset's attributes
df.describe()

# Print the list of columns in the dataset (features and target)
list(df.columns)

# Define the list of continuous feature names
features_continuous = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', \
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Define the target variable name
target = ['popularity']

# Define the path where results should be saved
save_path = './Results/Continuous'


#================================Pre-processing===============================
prompt_continue("Pre-processing")
print("\n================================Pre-processing=================================")

# Shuffle the dataset randomly, ensuring reproducibility with a set random seed
df_encoded = df.sample(frac=1, random_state=42)

# Create a dataset with only the specified features and target
Xy = df_encoded[features_continuous+target]

# Print the number of unique examples and unique data points in the dataset (which might be labelled differently)
print('Number of non-duplicated examples: '+str(Xy[~Xy.duplicated()==1].shape[0]))
print('Number of non-duplicated datapoints: '+str(Xy[features_continuous][~Xy[features_continuous].duplicated()==1].shape[0]))

# Removing examples that are equal in way they describe X but that are labelled differently
Xy = Xy[~Xy[features_continuous].duplicated()==1]

# Split the dataset into features (X) and target (y)
X = Xy[features_continuous]
y = Xy[target]


#=========================Training (Ridge Regression)=========================
prompt_continue("Training (Ridge Regression)")
print("\n=========================Training (Ridge Regression)=========================")

# Set the number of folds for cross-validation, the hyperparameters space, 
# the number sample sizes to consider for learning curve evaluation and the type
# of CV/hypertuning method 
K = 5
hyper_space = np.linspace(0, 15000, num=150)
samples = 25
type = 'nestedKFoldCV'

# Define the scalers (features transformation) to be used for the specified features
scalers = {
    ('duration_ms', 'loudness', 'tempo', 'popularity', 'energy', 'danceability', \
    'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence'): StandardScaler()
}

# Perform K-fold nested cross-validation using Ridge Regression (generating also the learning curve)
tuning_validation = expectedRiskEstimate(K, RidgeRegression, scalers=scalers, features=features_continuous, target=target, \
                                 evaluation=r2_score, verbose=True, visualize=True)
tuning_validation.nestedKFoldCV(X, y, hyper_space)
tuning_validation.save(save_path)
tuning_validation.save_predictors(save_path)
opt_alpha_no_scaling = tuning_validation.get_hyper_opt()

prompt_continue("Training (Ridge Regression: learning curve)")
print("\n==================Training (Ridge Regression: learning curve)==================")
tuning_validation.set_visualize(False)
tuning_validation.learningCurveCV(X, y, hyper_space, type, samples)


#=========================Model comparison====================================
prompt_continue("Model comparison")
print("\n=========================Model comparison=========================")


X_scaled = X 
y_scaled = y 

# Initializing the sklearn Ridge regression model
model = Ridge()

# Scaling the features and target using the provided scalers
for features, scaler in scalers.items():
    feat = list(features)
    if target[0] in features:
        feat.remove(target[0]) 
    X_scaled[feat] = scaler.fit_transform(X_scaled[feat])
    if target[0] in features:
        y_scaled = scaler.fit_transform(y_scaled)

# Defining a scorer based on R2 score
mse_scorer = make_scorer(r2_score)

# Performing cross-validation and obtaining the scores
scores = cross_val_score(model, X_scaled, y_scaled, cv=5, scoring=mse_scorer)

# Printing the cross-validation scores and their mean (expected risk estimate)
scores, scores.mean()

# Performing a validation curve to visualize the performance of the model for each hyperparameter
hyper_space = np.linspace(0, 15000, num=150)
train_scores_curve, test_scores_curve = validation_curve(model, X_scaled, y_scaled, param_name="alpha", param_range=hyper_space, cv=5, scoring=mse_scorer)

# Visualizing the validation curve
hyper_tuning_curve(hyper_space, train_scores_curve.T, test_scores_curve.T, method='Accuracy')

# Initializing grid search with the model, parameter grid, and scoring method
param_grid = {'alpha': hyper_space}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring=mse_scorer)
grid_search.fit(X_scaled, y_scaled)
cv_results = grid_search.cv_results_
mean_test_scores = cv_results['mean_test_score']
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv(save_path+'/grid_serach_results.csv', index=False)
df_to_latex_table(results_df, save_path+'/grid_serach_results.txt')


#=========================Training (Kernel ridge regression)====================
prompt_continue("Training (Kernel ridge regression)")
print("\n=========================Training (Kernel ridge regression)====================")

# Defining the save path for this phase
save_path = './Results/Kernel/Continuous'

# Defining cross-validation folds and hyperparameter space
K = 5
alphas = np.array([0.001, 0.1, 0.5, 1,  5, 10, 50, 100, 1000])
gammas = np.array([0.00001, 0.001, 0.01, 0.1, 1, 2, 4, 5, 10, 15, 20, 50, 100, 500, 4000])
hyper_space = np.array(list(product(alphas, gammas)))

# Define the scalers (features transformation) to be used for the specified features
scalers = {
    ('duration_ms', 'loudness', 'tempo', 'popularity', 'energy', 'danceability', \
    'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence'): StandardScaler()
}

# Reducing the dataset size
X_red=X[:4000]
y_red=y[:4000]

# Applying the expected risk estimation with hyperparameters tuning (best CV estimate) for kernel ridge regression
best_cv = expectedRiskEstimate(K, KernelRidgeRegression, scalers=scalers, features=features_continuous, \
                        target=target, evaluation=r2_score, dc_applied=False, verbose=True, visualize=True)
best_cv.bestCV(X_red, y_red, hyper_space)
best_cv.save_best(save_path)
best_cv.save_predictors(save_path)



#=========================Training (DC Kernel ridge regression)====================
prompt_continue("Training (DC Kernel ridge regression)")
print("\n=========================DC Training (Kernel ridge regression)====================")

# Defining the save path for this phase
save_path = './Results/Kernel/Continuous'

# Defining cross-validation folds and hyperparameter space
K = 5
hyper_space = np.array([[5, 10], [1, 20]])

# Define the scalers (features transformation) to be used for the specified features
scalers = {
    ('duration_ms', 'loudness', 'tempo', 'popularity', 'energy', 'danceability', \
    'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence'): StandardScaler()
}

# Reducing the dataset size
X_red=X[:83475]
y_red=y[:83475]

# Define the maximum allowed training size
sample_tollerance = 3200

# Perform expected risk estimate for DC Kernel Ridge
best_cv = expectedRiskEstimate(K, DCKRR, scalers=scalers, features=features_continuous, \
                        target=target, evaluation=r2_score, sample_tollerance=sample_tollerance, \
                        dc_applied=True, verbose=True, visualize=True)
best_cv.bestCV(X_red, y_red, hyper_space)
best_cv.save_best(save_path)
best_cv.save_predictors(save_path)


