import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import pickle


def hyper_tuning_curve(hyper_space, train_scores, validation_scores, method='Error'):
    """
    Plots validation curve (plt that shows scores as a function of the hyperparameters values).
    This validation curve is employed the nestedKFoldCV function
    
    Parameters:
    - hyper_space: the hyperparameter space.
    - train_scores: training scores.
    - validation_scores: validation/test scores.
    - method: 'Error' or 'Accuracy'.
    
    Displays:
    - A curve plot showing the relation between hyperparameters and the scores.
    """
    mean_training_score = np.mean(train_scores, 0)
    std_training_score = np.std(train_scores, 0)
    mean_validation_score = np.mean(validation_scores, 0)
    std_validation_score = np.std(validation_scores, 0)

    sns.set_context('poster')

    plt.ion()
    plt.figure()
    plt.plot(hyper_space, mean_training_score, label='Training '+method)
    plt.plot(hyper_space, mean_validation_score, label='Validation '+method)
    plt.legend()
    plt.xlabel('Regualization factor')
    plt.ylabel(method)
    plt.title("Validation Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def hyper_tuning_curve_best(hyper_space, train_scores, validation_scores, method='Error'):
    """
    Plots validation curve (plt that shows scores as a function of the hyperparameters values).
    This validation curve is employed the bestCV function.
    
    Parameters:
    - hyper_space: the hyperparameter space.
    - train_scores: training scores.
    - validation_scores: validation/test scores.
    - method: 'Error' or 'Accuracy'.
    
    Displays:
    - A curve plot showing the relation between hyperparameters and the scores.
    """
    sns.set_context('poster')
    plt.ion()
    plt.figure()
    plt.plot(hyper_space, train_scores, label='Training '+method)
    plt.plot(hyper_space, validation_scores, label='Validation '+method)
    plt.legend()
    plt.xlabel('Regualization factor')
    plt.ylabel(method)
    plt.title("Validation Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def hyper_tuning_curve_3D(hyper_space, train_scores, validation_scores, method='Error'):
    """
    Plots validation curve (plt that shows scores as a function of the hyperparameters values).
    This validation curve is employed the nestedKFoldCV function when the hyperparameter space has 
    two dimensions.
    
    Parameters:
    - hyper_space: the hyperparameter space.
    - train_scores: training scores.
    - validation_scores: validation/test scores.
    - method: 'Error' or 'Accuracy'.
    
    Displays:
    - A curve plot showing the relation between hyperparameters and the scores.
    """
    alpha_mesh, gamma_mesh = np.meshgrid(np.unique(hyper_space[:,0]), np.unique(hyper_space[:,1]))
    #training_scores_grid = train_scores.reshape(len(hyper_space), len(hyper_space))
    #validation_scores_grid = validation_scores.reshape(len(hyper_space), len(hyper_space))
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(alpha_mesh, gamma_mesh, train_scores.reshape(alpha_mesh.shape), cmap='viridis', label='Training Scores')
    ax.plot_surface(alpha_mesh, gamma_mesh, validation_scores.reshape(alpha_mesh.shape), cmap='plasma', label='Validation Scores')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Gamma')
    ax.set_zlabel(method)
    ax.set_title('3D Validation Curve')
    plt.show()
    with open('3d_kernel_valcurve.pickle', 'wb') as f:
        pickle.dump(fig, f)
    

def plot_learning_curve(train_sizes, train_scores, test_scores, method='Error'):
    """
    Plots learning curve for train and test scores against varying training set sizes.
    
    Parameters:
    - train_sizes: Sizes of the training set.
    - train_scores: Training scores.
    - test_scores: Test scores.
    - method: Metric to be plotted. Default is 'Error'.
    
    Displays:
    - A curve plot showing the learning progression with increasing training size.
    """
    sns.set_context('poster')
    plt.ion()
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training set size")
    plt.ylabel(method)
    plt.plot(train_sizes, train_scores, color="r", label="Training "+method)
    plt.plot(train_sizes, test_scores, color="g", label="Test "+method)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def plot_hyperparam_curve(ms, hyper_spaces):  
    """
    Plots which value is chosen, for each fold, as the training size varies.
    
    Parameters:
    - ms: Training set sizes.
    - hyper_spaces: Chosen hyperparameters for each model.
    
    Displays:
    - A scatter plot showing the hyperparameters chosen for each fold at each trainig size.
    """
    sns.set_context('poster')
    plt.ion()
    plt.figure()  # Imposta la dimensione della figura   
    plt.title("Hyperparameters Curve")
    for i, hyper in enumerate(hyper_spaces):
        c = 0.8 - 0.15*i
        s = 25 - 5*i
        #plt.scatter(ms, hyper, c=[c], cmap=cm.gray, s=point_size, label=f'Fold {i}')
        plt.plot(ms, hyper, 'o', markersize=s, color=(c, c, c), label=f'Fold {i}')
        #plt.plot(ms, hyper, 'o', label=f'Fold {i}')         
    plt.xlabel('Training set size')  # Etichetta dell'asse x
    plt.ylabel('Hyper values')  # Etichetta dell'asse y
    plt.legend()  # Mostra le legende per le righe  
    plt.grid(True)
    plt.show()  # Mostra il grafico



def plot_linear_predictor(X, y, w, size, caption=None):
    """
    Plots the linear predictor broken down into its d components (each subplot displays the predictor’s intercept 
    and j-th coefficient, with j=1,...,d,  revealing the relationships that a predictor attempts to capture between 
    each feature and the target.
    
    Parameters:
    - X: Feature matrix.
    - y: Target variable.
    - w: Weights of the linear predictor.
    - size: Number of examples to sample for plotting.
    - caption: Caption for the plot title.
    
    Displays:
    - Subplots showing each linear predictor component.
    """
    d = X.shape[1]  
    M = int(np.ceil(np.sqrt(d)))  
    N = int(np.ceil(d / M))  

    features = X.columns.tolist()

    fig, axes = plt.subplots(M, N, figsize=(15, 10))

    fig.suptitle('Linear Predictor (fold n.'+str(caption[0]+1)+', '+str(caption[1])+')')

    np.random.seed(42) 

    for i, ax in enumerate(axes.flat):
        if i < d:
            feature = X[features[i]].to_numpy()
            target = y.to_numpy()
            indices = np.random.choice(feature.shape[0], size=size, replace=False)
            sampled_feature = feature[indices]
            sampled_target = target[indices] 
            ax.scatter(sampled_feature, sampled_target, color='blue', alpha=0.5, label='Data Points')
            #ax.plot(feature, w[0] + w[i+1] * feature, color='red', label='Predictor')
            x1 = feature.min()
            y1 = w[0]
            x2 = feature.max()
            y2 = w[0] + x2 * w[i+1]
            ax.plot([x1, x2], [y1, y2], color='red')
            ax.set_xlabel(f'Feature {i+1}: '+features[i])
            ax.set_ylabel('Target: '+y.columns.tolist()[0])
            ax.legend()

        ax.set_xticks([])
        ax.set_yticks([])

    plt.ion()
    plt.tight_layout()
    plt.show()

def plot_histo(X, features, caption):
    """
    Plots histograms for the given features.
    
    Parameters:
    - X: DataFrame containing the features.
    - features: List of feature names to plot.
    - caption: Title caption for the plot.
    
    Displays:
    - Histograms for each feature.
    """
    plt.ion()
    if len(features)/2 > 2:
        if len(features)%2 == 0:
            fig, axes = plt.subplots(nrows=2, ncols=int(len(features)/2), figsize=(18, 6))
        else: 
            fig, axes = plt.subplots(nrows=2, ncols=int(len(features)/2)+1, figsize=(18, 6))
    else: 
        if len(features)==1:
            fig, ax = plt.subplots(figsize=(18, 6))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=int(len(features)), figsize=(18, 6))
    fig.suptitle(caption)
    if len(features)==1:
        sns.histplot(data=X[features], x=features[0])
    else:
        axes = axes.flatten()
        for i, col in enumerate(features):
            sns.histplot(data=X[features], x=col, ax=axes[i])
    plt.tight_layout()
    plt.show()

def plot_correlation_full(df):
    """
    Plots full correlation heatmap of the DataFrame's features.
    
    Parameters:
    - df: DataFrame to compute and plot correlations for.
    
    Displays:
    - A heatmap showing linear correlation between features.
    """
    plt.ion()
    corr = np.abs(df.corr())
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.color_palette("Reds")
    sns.heatmap(corr, cmap=cmap, square=True)
    plt.title('Linear correlation between features (absolute values)')
    plt.tight_layout()
    plt.show()
    corr = corr[['popularity']].sort_values(by='popularity', ascending=False)
    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(corr, annot=True, cmap='Reds')
    heatmap.set_title('Target-correlated features', fontdict={'fontsize':18}, pad=16)


def find_nearest_divisor(N, v):
    """
    Finds the divisor of N that is nearest to a given value v.
    
    Parameters:
    - N: The number for which divisors are considered.
    - v: The value to which we want to find the nearest divisor.
    
    Returns:
    - Nearest divisor of N to the given value v.
    """
    nearest_divisor = None
    min_difference = float('inf')
    for divisor in range(1, N + 1):
        if N % divisor == 0:
            difference = abs(v - divisor)
            if difference < min_difference:
                min_difference = difference
                nearest_divisor = divisor

    return nearest_divisor

def df_to_latex_table(df, path):
    """
    Converts a DataFrame to a LaTeX table and writes it to a file.
    
    Parameters:
    - df: DataFrame to be converted.
    - path: File path to save the LaTeX table.
    
    Saves:
    - LaTeX formatted table to the given path.
    """
    latex_table = df.to_latex(index=False, escape=False)
    latex_table = latex_table.replace('\n', '')
    with open(path, 'w') as file:
        file.write(latex_table)

