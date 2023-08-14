import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import product
from sklearn.preprocessing import *
from sklearn.metrics import *
import category_encoders as ce
import sklearn
sklearn.set_config(transform_output="pandas")
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

#import code;
#code.interact(local=locals())


"""
RIDGE REGRESSION AND KERNEL RIDGE REGRESSION ON NUMERICAL AND CATEGORICAL FEATURES

OMAR GHEZZI matr. 984352

MAIN: run the data cleaning, pre-processing to the whole dataset. Apply 5-fold
nested cross validation to ridge regression on the resulting pre-processed dataset.
Apply 5-fold best CV estimate to kernel ridge regression (KRR) and DCKRR on
4000 and 157185 examples, respectively.


#####################################################################################################################
#                                                                                                                   #
#       explicit        ->      binary {0,1}                                                                        #
#       key             ->      ordinal (qualitative)       [-1,11]     [-1 = Missing, 0 = C, 1 = C#/Db,...]        #  
#       mode            ->      binary {0,1}                                                                        #
#       time_signature  ->      ordinal (qualitative)       [3,7]       [3 = 3/4, 4 = 4/4, ..., 7 = 7/4]            #
#       track_genre     ->      categorical                 (String)    114 unique values                           #
#       track_name      ->      categorical                 (String)    73609 unique values                         #
#       album_name      ->      categorical                 (String)    46590 unique values                         #
#       artists         ->      categorical                 (String)    31438 unique values                         #
#                                                                                                                   #
#####################################################################################################################

"""

#==============================================================================
#==               1-B)CONTINUOUS + BINARY +CATEGORIAL FEATURE TRAINING       ==
#==============================================================================

#================================DATA LOADING=================================
print("\n================================DATA LOADING=================================")

# Read dataset from CSV file into a DataFrame
df = pd.read_csv('./data/dataset.csv')

# Remove the first two columns ('Unnamed: 0' and 'track_id') from the DataFrame
df = df.drop(columns=df.columns[0:2])

# Print the header of the dataset
df.head()

# Print a description of the dataset
df.describe()

# Print the list of the attributes (plus the target)
list(df.columns)

# Define lists of features based on their type
features_continuous = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', \
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
features_categorical = ['key', 'time_signature', 'track_genre', 'album_name', 'track_name']
features_binary = ['explicit', 'mode']
features_multi_values = ['artists']
target = ['popularity']

# Define the path where results will be saved
save_path = './Results/Categorical_alt'


#================================DATA CLEANING=================================
print("\n================================DATA CLEANING=================================")

# Count missing 'key' values and drop rows with missing values
count_missing_key = (df['key'] == -1).sum() 
if df.isna().sum().any():
    print(df[df.isna().any(axis=1)])
    df = df.dropna()

# Handle multivalued features: add a different row for each artist in a featuring
df[features_multi_values] = df[features_multi_values].apply(lambda x: x.str.split(';'))
df = df.explode(features_multi_values)

# Handle the $ character for the artist 'Bad$$'
df[features_multi_values] = df[features_multi_values].apply(lambda x: x.str.replace('\$\$','\\$\\$'))

# Encode the 'explicit' feature as 0-1 instead as False-True
df["explicit"] = df["explicit"].astype(int)

# Random shuffling of the dataset
df_encoded = df.sample(frac=1, random_state=42)

# Select relevant features and target variable from the shuffled data
total_features=features_continuous+features_categorical+features_multi_values+features_binary
Xy = df_encoded[total_features+target]
print('Number of non-duplicated examples: '+str(Xy[~Xy.duplicated()==1].shape[0]))
print('Number of non-duplicated datapoints: '+str(Xy[total_features][~Xy[total_features].duplicated()==1].shape[0]))

# Removing examples that are equal in way they describe X but that are labelled differently
Xy = Xy[~Xy[total_features].duplicated()==1]

# Split the dataset into features (X) and target (y)
X = Xy[total_features]
y = Xy[target]


#--------------------------Training (Ridge Regression): target + count encoding-------------------------
prompt_continue("Training (Ridge Regression): target + count encoding")
print("\n=========================Training (Ridge Regression): target + count encoding=========================")

# Set the number of folds for cross-validation, the hyperparameters space, 
# the number sample sizes to consider for learning curve evaluation and the type
# of CV/hypertuning method
K = 5
hyper_space = np.linspace(0, 15000, num=150)
samples = 25
type = 'nestedKFoldCV'

# Define encoders (features encoding) for the categorical variables
encoders = {
    ('track_genre'): TargetEncoding(target_col=target, encoding_type='mean', smoothing=5, default_shrink=3),
    ('artists', 'album_name', 'track_name'): ce.CountEncoder()
}

# Define the scalers (features transformation) to be used for the specified features
scalers = {
    ('duration_ms', 'loudness', 'tempo', 'popularity', 'energy', 'danceability', \
    'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence', \
     'artists', 'time_signature', 'key', 'track_genre', 'album_name', 'track_name'): StandardScaler()
}

# Perform K-fold nested cross-validation using Ridge Regression (generating also the learning curve)
features = features_continuous+features_categorical+features_multi_values+features_binary
tuning_validation = expectedRiskEstimate(K, RidgeRegression, scalers=scalers, encoders=encoders, features=features, \
                       target=target, evaluation=r2_score, verbose=True, visualize=True)
tuning_validation.nestedKFoldCV(X, y, hyper_space)
tuning_validation.save(save_path)
tuning_validation.save_predictors(save_path)
optimal_alpha = tuning_validation.get_hyper_opt()

prompt_continue("Training (Ridge Regression: learning curve)")
print("\n==================Training (Ridge Regression: learning curve)==================")
tuning_validation.set_visualize(False)
tuning_validation.learningCurveCV(X, y, hyper_space, type, samples)


#--------------------------Training (Ridge Regression): one-hot + count encoding-------------------------
prompt_continue("Training (Ridge Regression): one-hot + count encoding")
print("\n=========================Training (Ridge Regression): one-hot + count encoding=========================")

# Set the number of folds for cross-validation, the hyperparameters space, 
# the number sample sizes to consider for learning curve evaluation and the type
# of CV/hypertuning method
K = 5
hyper_space = np.linspace(0, 15000, num=150)
samples = 25
type = 'nestedKFoldCV'

# Define encoders (features encoding) for the categorical variables
encoders = {
    ('track_genre'): OneHotEncoder(sparse_output=False),
    ('artists', 'album_name', 'track_name'): ce.CountEncoder()
}

# Define the scalers (features transformation) to be used for the specified features
scalers = {
    ('duration_ms', 'loudness', 'tempo', 'popularity', 'energy', 'danceability', \
    'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence', \
     'artists', 'time_signature', 'key', 'album_name', 'track_name'): StandardScaler()
}

# Perform K-fold nested cross-validation using Ridge Regression (generating also the learning curve)
features = features_continuous+features_categorical+features_multi_values+features_binary
tuning_validation = expectedRiskEstimate(K, RidgeRegression, scalers=scalers, encoders=encoders, features=features, \
                       target=target, evaluation=r2_score, verbose=True, visualize=True)
tuning_validation.nestedKFoldCV(X, y, hyper_space)
tuning_validation.save(save_path)
tuning_validation.save_predictors(save_path)
optimal_alpha = tuning_validation.get_hyper_opt()

prompt_continue("Training (Ridge Regression: learning curve)")
print("\n==================Training (Ridge Regression: learning curve)==================")
tuning_validation.set_visualize(False)
tuning_validation.learningCurveCV(X, y, hyper_space, type, samples)


#--------------------------Training (Ridge Regression): one-hot + base2encoding-------------------------
prompt_continue("Training (Ridge Regression): one-hot + base2encoding")
print("\n=========================Training (Ridge Regression): one-hot + base2encoding=========================")

# Set the number of folds for cross-validation, the hyperparameters space, 
# the number sample sizes to consider for learning curve evaluation and the type
# of CV/hypertuning method
K = 5
hyper_space = np.linspace(0, 15000, num=150)
samples = 20
type = 'nestedKFoldCV'

# Define encoders (features encoding) for the categorical variables
encoders = {
    ('track_genre'): OneHotEncoder(sparse_output=False),
    ('artists', 'album_name', 'track_name'): ce.HashingEncoder(n_components=50)
}

# Define the scalers (features transformation) to be used for the specified features
scalers = {
    ('duration_ms', 'loudness', 'tempo', 'popularity', 'energy', 'danceability', \
    'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence', \
    'time_signature', 'key', 'artists', 'album_name', 'track_name'): StandardScaler()
}

# Perform K-fold nested cross-validation using Ridge Regression (generating also the learning curve)
features = features_continuous+features_categorical+features_multi_values+features_binary
tuning_validation = expectedRiskEstimate(K, RidgeRegression, scalers=scalers, encoders=encoders, features=features, \
                       target=target, evaluation=r2_score, verbose=True, visualize=True)
tuning_validation.nestedKFoldCV(X, y, hyper_space)
tuning_validation.save(save_path)
tuning_validation.save_predictors(save_path)
optimal_alpha = tuning_validation.get_hyper_opt()

prompt_continue("Training (Ridge Regression: learning curve)")
print("\n==================Training (Ridge Regression: learning curve)==================")
tuning_validation.set_visualize(False)
tuning_validation.learningCurveCV(X, y, hyper_space, type, samples)


#=========================Training (Kernel ridge regression)====================
prompt_continue("Training (Kernel ridge regression)")
print("\n=========================Training (Kernel ridge regression)====================")

save_path = './Results/Kernel/Categorical'

# Defining cross-validation folds and hyperparameter space
K = 5
alphas = np.array([0.001, 0.1, 0.5, 1,  5, 10, 50, 100, 1000])
gammas = np.array([0.00001, 0.001, 0.01, 0.1, 1, 2, 4, 5, 10, 15, 20, 50, 100, 500, 4000])
hyper_space = np.array(list(product(alphas, gammas)))

# Define encoders (features encoding) for the categorical variables
encoders = {
    ('track_genre'): TargetEncoding(target_col=target, encoding_type='mean', smoothing=5, default_shrink=3),
    ('artists', 'album_name', 'track_name'): ce.CountEncoder()
}

# Define the scalers (features transformation) to be used for the specified features
scalers = {
    ('duration_ms', 'loudness', 'tempo', 'popularity', 'energy', 'danceability', \
    'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence', \
     'artists', 'track_genre', 'time_signature', 'key', 'album_name', 'track_name'): StandardScaler()
}

# Reducing the dataset size
X_red=X[:4000]
y_red=y[:4000]

# Applying the expected risk estimation with hyperparameters tuning (best CV estimate) for kernel ridge regression
features = features_continuous+features_categorical+features_multi_values+features_binary
best_cv = expectedRiskEstimate(K, KernelRidgeRegression, scalers=scalers, encoders=encoders, features=features, \
                        target=target, evaluation=r2_score, dc_applied=False, verbose=True, visualize=True)
best_cv.bestCV(X_red, y_red, hyper_space)
best_cv.save_best(save_path)
best_cv.save_predictors(save_path)


#=========================Training (DC Kernel ridge regression)====================
prompt_continue("Training (DC Kernel ridge regression)")
print("\n=========================DC Training (Kernel ridge regression)====================")

save_path = './Results/Kernel/Continuous'

# Defining cross-validation folds and hyperparameter space
K = 5
hyper_space = np.array([[5, 15]])

# Define encoders (features encoding) for the categorical variables
encoders = {
    ('track_genre'): TargetEncoding(target_col=target, encoding_type='mean', smoothing=5, default_shrink=3),
    ('artists', 'album_name', 'track_name'): ce.CountEncoder()
}

# Define the scalers (features transformation) to be used for the specified features
scalers = {
    ('duration_ms', 'loudness', 'tempo', 'popularity', 'energy', 'danceability', \
    'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence', \
     'artists', 'track_genre', 'time_signature', 'key', 'album_name', 'track_name'): StandardScaler()
}

# Reducing the dataset size
X_red=X[:157185]
y_red=y[:157185]

# Applying the expected risk estimation with hyperparameters tuning (best CV estimate) for kernel ridge regression
sample_tollerance = 7000
features = features_continuous+features_categorical+features_multi_values+features_binary
best_cv = expectedRiskEstimate(K, DCKRR, scalers=scalers, encoders=encoders, features=features, \
                        target=target, evaluation=r2_score, sample_tollerance=sample_tollerance, \
                        dc_applied=True, verbose=True, visualize=True)
best_cv.bestCV(X_red, y_red, hyper_space)
best_cv.save_best(save_path)
best_cv.save_predictors(save_path)
