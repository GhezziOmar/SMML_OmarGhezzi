import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from TargetEncoding import TargetEncoding
from functions import *
from DCKRR import DCKRR
from KernelRidgeRegression import KernelRidgeRegression
import inspect

"""
This class provides the functionalites to estimate the expected risk of a given predictor,
while performing hyperparameters tuning. It also provides features transformation and feature encoding,
as a first step. The implemented techniques are K-fold nested cross validation (nestedKFoldCV) and
K-fold best CV (bestCV) wich is computationally more efficient but statistically less rigorous than 
K-fold nested cross validation.
"""


class expectedRiskEstimate:

    def __init__(self, K, model, scalers=None, encoders=None, features=None, target=None, evaluation=mean_squared_error, sample_tollerance=4000, dc_applied=False, verbose=False, visualize=False):
        self.K = K                                          # Number of cross-validation folds
        self.model = model                                  # The regression model to be evaluated (RidgeRegression, KernelRidgeRegression or DCKRR)
        self.scalers = scalers                              # Scalers for features transforming
        self.encoders = encoders                            # Encoders for categorical variables
        self.target = target                                # Target variable
        self.features = features                            # Full list of feature names
        self.evaluation = evaluation                        # The evaluation metric (mean squared error or r2_score)
        self.val_training_errors = []                       # Validation training errors/scores for each fold
        self.validation_errors = []                         # Validation errors/scores for each fold
        self.training_errors = []                           # Training errors/scores for each fold
        self.test_errors = []                               # Test errors/scores for each fold
        self.mean_training_error = 0                        # Average training error/score
        self.risk_esimate = 0                               # Risk estimate (average test error/score)
        self.hyper_opt = []                                 # Optimal hyperparameters tuple
        self.predictors = []                                # Optimal predictors for each fold
        self.learningC_train = []                           # Learning curve - training errors/scores
        self.learningC_test = []                            # Learning curve - test errors/scores
        self.learningC_hyper = []                           # Learning curve - optimal hyperparameters
        self.learningC_model = []                           # Learning curve - optimal predictors
        self.learningC_ms = []                              # Learning curve - training sizes
        self.sample_tollerance = sample_tollerance          # Maximum training sizes (if model is DCKRR)
        self.dc_applied = dc_applied                        # Flag to indicate if model is DCKRR
        self.verbose = verbose                              # Flag for verbose output
        self.visualize = visualize                          # Flag for visualizing plots
        self.is_first_call = True                           # Flag to track if it's the first call to the function
    

    def nestedKFoldCV(self, X, y, hyper_space):
        """
        Implements the nested K-Fold cross-validation technique for hyperparameter tuning and expected risk estimate.
        
        Parameters:
        - X: Feature matrix
        - y: Target variable
        - hyper_space: Hyperparameters space
        
        Returns:
        - mean_training_error: Average training error
        - risk_estimate: Expected risk estimate
        - hyper_opt: Optimal hyperparameters
        - predictors: Predictors for each fold
        - m: Number of training samples used in each fold
        """

        self.val_training_errors = []
        self.validation_errors = []
        self.training_errors = []
        self.test_errors = []
        self.hyper_opt = []
        self.predictors = []

        sample_size = X.shape[0]
        fold_size = sample_size // self.K
        m = sample_size * (self.K-1) // self.K
        val_size = int(0.2 * m)

        X_folds = np.array_split(X, self.K)
        y_folds = np.array_split(y, self.K)

        for i in range(self.K):

            if self.verbose : 
                print("\n==========================Processing "+str(i)+"-th Fold==========================")
            X_train_full = pd.concat(X_folds[:i]+X_folds[i+1:]).copy()
            y_train_full = pd.concat(y_folds[:i]+y_folds[i+1:]).copy()
           
            X_train = X_train_full[:m - val_size].copy() 
            X_validation = X_train_full[m - val_size:].copy() 
            X_test = X_folds[i].copy()
            y_train = y_train_full[:m - val_size].copy() 
            y_validation = y_train_full[m - val_size:].copy() 
            y_test = y_folds[i].copy()

            new_columns_print = []

            if self.encoders is not None :
                for features, encoder in self.encoders.items():
                    if isinstance(features, str):
                        encoder = self.encoders[features]
                        features = [features]
                    else:
                        features = list(features)
                    if isinstance(encoder, TargetEncoding):                  
                        X_train[features] = encoder.fit_transform(pd.concat([X_train, y_train], axis=1), encoding_cols=features)
                        X_validation[features] = encoder.transform(X_validation, encoding_cols=features)
                        X_train_full[features] = encoder.fit_transform(pd.concat([X_train_full, y_train_full], axis=1), encoding_cols=features)
                        X_test[features] = encoder.transform(X_test, encoding_cols=features)
                    elif isinstance(encoder, OneHotEncoder) or (isinstance(encoder, ce.BaseNEncoder) and encoder.get_params()['base']==2):
                        X_train = X_train.drop(columns=features).assign(**encoder.fit_transform(X_train[features]))
                        X_validation = X_validation.drop(columns=features).assign(**encoder.transform(X_validation[features]))
                        X_train_full = X_train_full.drop(columns=features).assign(**encoder.fit_transform(X_train_full[features]))
                        X_test = X_test.drop(columns=features).assign(**encoder.transform(X_test[features]))
                        self.features = list(X_train.columns)                        
                    elif isinstance(encoder, ce.WOEEncoder):
                        X_train[features] = encoder.fit_transform(X_train[features], y_train)
                        X_validation[features] = encoder.transform(X_validation[features], y_validation)
                        X_train_full[features] = encoder.fit_transform(X_train_full[features], y_train_full)
                        X_test[features] = encoder.transform(X_test[features], y_test)
                    else:
                        transform_train = encoder.fit_transform(X_train[features])
                        X_train = X_train.drop(columns=features).assign(**transform_train)
                        X_validation = X_validation.drop(columns=features).assign(**encoder.transform(X_validation[features]))
                        X_train_full = X_train_full.drop(columns=features).assign(**encoder.fit_transform(X_train_full[features]))
                        X_test = X_test.drop(columns=features).assign(**encoder.transform(X_test[features]))
                        self.features = list(X_train.columns) 
                        new_columns = transform_train.columns.tolist()
                        new_columns_print += new_columns
                        if self.scalers is not None and i==0 and self.is_first_call:
                            old_columns = list(list(self.scalers.keys())[0])
                            value = next(iter(self.scalers.values()))  
                            del self.scalers[tuple(old_columns)] 
                            for feature in features:
                                old_columns.remove(feature)  
                            comb_columns = old_columns + new_columns 
                            self.scalers[tuple(comb_columns)] = value
            
            if(inspect.currentframe().f_back.f_code.co_name == 'learningCurveCV'):
                self.is_first_call = False
            
            if self.visualize and i==0:
                if self.encoders is None:
                    plot_histo(pd.concat([X_train, y_train], axis=1), self.features+self.target, caption='Features histograms')
                elif not isinstance(encoder, OneHotEncoder) and not isinstance(encoder, ce.BaseNEncoder):
                    plot_histo(X_train[new_columns_print], new_columns_print, caption='Features histograms (categorical)')

            
            if self.scalers is not None:
                for features, scaler in self.scalers.items():
                    feat = list(features)
                    if self.target[0] in features:
                        feat.remove(self.target[0]) 
                    X_train[feat] = scaler.fit_transform(X_train[feat])
                    X_validation[feat] = scaler.transform(X_validation[feat])
                    if self.target[0] in features:
                        y_train = scaler.fit_transform(y_train)
                        y_validation = scaler.transform(y_validation)
                    X_train_full[feat] = scaler.fit_transform(X_train_full[feat])
                    X_test[feat] = scaler.transform(X_test[feat])
                    if self.target[0] in features:
                        y_train_full = scaler.fit_transform(y_train_full)
                        y_test = scaler.transform(y_test)
                      
            if self.visualize and i==0:
                if self.encoders is None:
                    plot_histo(pd.concat([X_train, y_train], axis=1), self.features+self.target, caption='Encoded and scaled features histograms')
                elif not isinstance(encoder, OneHotEncoder) and not isinstance(encoder, ce.BaseNEncoder):
                    plot_histo(X_train[new_columns_print], new_columns_print, caption='Encoded and scaled features histograms (categorical)')
          
            if self.dc_applied :
                num_subsets_candidate_train = int(X_train.shape[0]/self.sample_tollerance)
                num_subsests_train = find_nearest_divisor(X_train.shape[0], num_subsets_candidate_train)
                num_subsets_candidate_train_full = int(X_train_full.shape[0]/self.sample_tollerance)
                num_subsests_train_full = find_nearest_divisor(X_train_full.shape[0], num_subsets_candidate_train_full)

            val_training_scores = []
            validation_scores = []
            inter_val_training_scores = []
            inter_test_scores = []

            for j, alpha in enumerate(hyper_space) : 
                if self.verbose :
                    print("\n----------------------Validate  "+str(i)+"-th Fold with regularization: "+str(alpha)+" -------------------------")
                
                model = self.model(alpha=alpha)
                if self.dc_applied :
                    model.set_num_subsets(num_subsests_train)
                    model.set_verbose(self.verbose)
                model.fit(X_train, y_train)
                val_training_scores.append(model.score(X_train, y_train, method=self.evaluation))
                validation_scores.append(model.score(X_validation, y_validation, method=self.evaluation)) 

                #---------additional part for plots-------------
                model = self.model(alpha=alpha)
                if self.dc_applied :
                    model.set_num_subsets(num_subsests_train_full)
                    model.set_verbose(self.verbose)
                model.fit(X_train_full, y_train_full)
                inter_val_training_scores.append(model.score(X_train_full, y_train_full, method=self.evaluation))
                inter_test_scores.append(model.score(X_test, y_test, method=self.evaluation)) 

            self.val_training_errors.append(inter_val_training_scores)
            self.validation_errors.append(inter_test_scores)
            
            if self.evaluation == mean_squared_error:
                self.hyper_opt.append(hyper_space[np.argmin(validation_scores)])
            elif self.evaluation == r2_score:
                self.hyper_opt.append(hyper_space[np.argmax(validation_scores)])
            
            if self.verbose :
                print("\n----------------------Test  "+str(i)+"-th Fold with regularization: "+str(self.hyper_opt[i])+" -------------------------")
                       
            model_opt = self.model(alpha=self.hyper_opt[i])
            if self.dc_applied :
                    model_opt.set_num_subsets(num_subsests_train_full)
                    model_opt.set_verbose(self.verbose)
            model_opt.fit(X_train_full, y_train_full)
            self.predictors.append(model_opt.get_w())
            self.training_errors.append(model_opt.score(X_train_full, y_train_full, method=self.evaluation))
            self.test_errors.append(model_opt.score(X_test, y_test, method=self.evaluation))

            if self.visualize and hyper_space.ndim==1:
                #plot_linear_predictor(X_train_full, y_train_full, model_opt.get_w(), caption=[i, 'Training'])
                plot_linear_predictor(X_test, y_test, model_opt.get_w(), fold_size//16, caption=[i, 'Test'])
        
        #import code;
        #code.interact(local=locals())
        
        if self.visualize and hyper_space.ndim==1:
            if self.evaluation == mean_squared_error:
                    hyper_tuning_curve(hyper_space, np.array(self.val_training_errors), np.array(self.validation_errors), 'Error')
            elif self.evaluation == r2_score:
                    hyper_tuning_curve(hyper_space, np.array(self.val_training_errors), np.array(self.validation_errors), 'Accuracy')
        elif self.visualize and hyper_space.ndim==2:
            if self.evaluation == mean_squared_error:
                hyper_tuning_curve_3D(hyper_space, np.array(self.val_training_errors), np.array(self.validation_errors), method='Error')
            elif self.evaluation == r2_score:
                hyper_tuning_curve_3D(hyper_space, np.array(self.val_training_errors), np.array(self.validation_errors), method='Accuracy')


        self.mean_training_error = sum(self.training_errors)/self.K
        self.risk_esimate = sum(self.test_errors)/self.K
        return self.mean_training_error, self.risk_esimate, self.hyper_opt, self.predictors, m


    
    
    def bestCV(self, X, y, hyper_space):
        """
        Implements a less demanding and less rigorous rivariant of hyperparameters tuning + expected risk estimate.

        Parameters:
        - X: Feature matrix
        - y: Target variable
        - hyper_space: Possible hyperparameters to test

        Returns:
        - mean_training_error: Average training error
        - risk_estimate: Expected risk estimate
        - hyper_opt: Optimal hyperparameters
        - predictors: Model predictors for each fold
        - m: Number of training samples used in each fold
        """

        self.training_errors = []
        self.test_errors = []

        sample_size = X.shape[0]
        fold_size = sample_size // self.K
        m = sample_size * (self.K-1) // self.K

        X_folds = np.array_split(X, self.K)
        y_folds = np.array_split(y, self.K)

        self.hyper_opt = 0

        X_train_full = []
        y_train_full = []
        X_test = []
        y_test = []

        num_subsests_train_full = []

        for i in range(self.K):

            if self.verbose : 
                print("\n==========================Create "+str(i)+"-th Fold==========================")

            X_train_full.append(pd.concat(X_folds[:i]+X_folds[i+1:]).copy())
            y_train_full.append(pd.concat(y_folds[:i]+y_folds[i+1:]).copy())       
            X_test.append(X_folds[i].copy()) 
            y_test.append(y_folds[i].copy())

        for i in range(self.K):

            if self.verbose : 
                print("\n==========================Pre-processing "+str(i)+"-th Fold==========================")
            
            new_columns_print = []

            if self.encoders is not None :
                for features, encoder in self.encoders.items():
                    if isinstance(features, str):
                        encoder = self.encoders[features]
                        features = [features]
                    else:
                        features = list(features)
                    if isinstance(encoder, TargetEncoding):                  
                        X_train_full[i][features] = encoder.fit_transform(pd.concat([X_train_full[i], y_train_full[i]], axis=1), encoding_cols=features)
                        X_test[i][features] = encoder.transform(X_test[i], encoding_cols=features)
                    elif isinstance(encoder, OneHotEncoder) or (isinstance(encoder, ce.BaseNEncoder) and encoder.get_params()['base']==2):
                        X_train_full[i] = X_train_full[i].drop(columns=features).assign(**encoder.fit_transform(X_train_full[i][features]))
                        X_test[i] = X_test[i].drop(columns=features).assign(**encoder.transform(X_test[i][features]))
                        self.features = list(X_train_full[i].columns)                        
                    elif isinstance(encoder, ce.WOEEncoder):
                        X_train_full[i][features] = encoder.fit_transform(X_train_full[i][features], y_train_full[i])
                        X_test[i][features] = encoder.transform(X_test[i][features], y_test[i])
                    else:
                        transform_train = encoder.fit_transform(X_train_full[i][features])
                        X_train_full[i] = X_train_full[i].drop(columns=features).assign(**encoder.fit_transform(X_train_full[i][features]))
                        X_test[i] = X_test[i].drop(columns=features).assign(**encoder.transform(X_test[i][features]))
                        self.features = list(X_train_full[i].columns) 
                        new_columns = transform_train.columns.tolist()
                        new_columns_print += new_columns
                        if self.scalers is not None and i==0 and self.is_first_call:
                            old_columns = list(list(self.scalers.keys())[0])
                            value = next(iter(self.scalers.values()))  
                            del self.scalers[tuple(old_columns)] 
                            for feature in features:
                                old_columns.remove(feature)  
                            comb_columns = old_columns + new_columns 
                            self.scalers[tuple(comb_columns)] = value
            
            if(inspect.currentframe().f_back.f_code.co_name == 'learningCurveCV'):
                self.is_first_call = False
            
            if self.visualize and i==0:
                if self.encoders is None:
                    plot_histo(pd.concat([X_train_full[i], y_train_full[i]], axis=1), self.features+self.target, caption='Features histograms')
                elif not isinstance(encoder, OneHotEncoder) and not isinstance(encoder, ce.BaseNEncoder):
                    plot_histo(X_train_full[i][new_columns_print], new_columns_print, caption='Features histograms (categorical)')

            
            if self.scalers is not None:
                for features, scaler in self.scalers.items():
                    feat = list(features)
                    if self.target[0] in features:
                        feat.remove(self.target[0]) 
                    X_train_full[i][feat] = scaler.fit_transform(X_train_full[i][feat])
                    X_test[i][feat] = scaler.transform(X_test[i][feat])
                    if self.target[0] in features:
                        y_train_full[i] = scaler.fit_transform(y_train_full[i])
                        y_test[i] = scaler.transform(y_test[i])
                      
            if self.visualize and i==0:
                if self.encoders is None:
                    plot_histo(pd.concat([X_train_full[i], y_train_full[i]], axis=1), self.features+self.target, caption='Encoded and scaled features histograms')
                elif not isinstance(encoder, OneHotEncoder) and not isinstance(encoder, ce.BaseNEncoder):
                    plot_histo(X_train_full[i][new_columns_print], new_columns_print, caption='Encoded and scaled features histograms (categorical)')
        
        models = []

        for j, alpha in enumerate(hyper_space) : 

            inter_training_scores = []
            inter_test_scores = []
            inter_model = [] 

            for i in range(self.K):

                if self.verbose :
                    print("\n----------------------Test  "+str(i)+"-th Fold with regularization: "+str(alpha)+" -------------------------")
                
                model = self.model(alpha=alpha)
                if self.dc_applied and isinstance(model, DCKRR):
                        num_subsets_candidate_train_full = int(X_train_full[i].shape[0]/self.sample_tollerance)
                        num_subsests_train_full.append(find_nearest_divisor(X_train_full[i].shape[0], num_subsets_candidate_train_full))
                        model.set_num_subsets(num_subsests_train_full[i])
                        model.set_verbose(self.verbose)
                if self.dc_applied and isinstance(model, KernelRidgeRegression):
                    model.set_apply_nystrom(True)
                model.fit(X_train_full[i], y_train_full[i])
                inter_model.append(model.get_w())
                inter_training_scores.append(model.score(X_train_full[i], y_train_full[i], method=self.evaluation))
                inter_test_scores.append(model.score(X_test[i], y_test[i], method=self.evaluation))
            
            models.append(inter_model)
            self.training_errors.append(sum(inter_training_scores)/self.K)
            self.test_errors.append(sum(inter_test_scores)/self.K)
        
        if self.evaluation == mean_squared_error:
            self.hyper_opt = hyper_space[np.argmin(self.test_errors)]
            self.predictors = models[np.argmin(self.test_errors)]
            self.mean_training_error = self.training_errors[np.argmin(self.test_errors)]
            self.risk_esimate = np.min(self.test_errors)
        elif self.evaluation == r2_score:
            self.hyper_opt = hyper_space[np.argmax(self.test_errors)]
            self.predictors = models[np.argmax(self.test_errors)]
            self.mean_training_error = self.training_errors[np.argmax(self.test_errors)]
            self.risk_esimate = np.max(self.test_errors)
        
        if self.visualize and hyper_space.ndim==1:
            plot_linear_predictor(X_test[0], y_test[0], self.predictors[0], fold_size//16, caption=[0, 'Test']) 
        
        if self.visualize and hyper_space.ndim==1:
            if self.evaluation == mean_squared_error:
                    hyper_tuning_curve_best(hyper_space, np.array(self.training_errors), np.array(self.test_errors), 'Error')
            elif self.evaluation == r2_score:
                    hyper_tuning_curve_best(hyper_space, np.array(self.training_errors), np.array(self.test_errors), 'Accuracy')
        elif self.visualize and hyper_space.ndim==2:
            if isinstance(model, KernelRidgeRegression):
                unique_gammas = np.unique(hyper_space[:, 1], return_index=False)
                unique_alphas = np.unique(hyper_space[:,0], return_index=False)
                alphas = []
                trains = []
                tests = []
                opt_trains = []
                opt_tests = []
                opt_alpha = []
                for i, gamma in enumerate(unique_gammas):
                    indices = np.where(hyper_space[:, 1] == gamma)[0]
                    trains.append(np.array(self.training_errors)[indices])
                    tests.append(np.array(self.test_errors)[indices])
                    alphas.append(unique_alphas)
                    if self.evaluation == mean_squared_error:
                        opt_trains.append(np.min(trains[i]))
                        opt_tests.append(np.min(tests[i]))
                        opt_alpha.append(unique_alphas[np.argmin(tests[i])])
                        hyper_tuning_curve_best(alphas[i], trains[i], tests[i], 'Error')
                    elif self.evaluation == r2_score:
                        opt_trains.append(np.max(trains[i]))
                        opt_tests.append(np.max(tests[i]))
                        opt_alpha.append(unique_alphas[np.argmax(tests[i])])
                        hyper_tuning_curve_best(alphas[i], trains[i], tests[i], 'Accuracy')
                results = pd.DataFrame()
                results["Gamma"] = unique_gammas
                results["Alpha"] = opt_alpha
                results["Alphas"] = alphas
                if self.evaluation == mean_squared_error:
                    results["Training_errors"] = trains
                    results["Test_errors"] = tests
                    results["Exp. training error estimate"] = opt_trains
                    results["Risk estimate (Exp. stat. risk estimate)"] = opt_tests
                if self.evaluation == r2_score:
                    results["Training_accuracy"] = trains
                    results["Test_accuracy"] = tests
                    results["Exp. training acc. estimate"] =opt_trains
                    results["Accuracy estimate (Exp. stat. acc. estimate)"] = opt_tests
                print(results)
                results.to_csv('./Kernel_results.csv', index=False)
                df_to_latex_table(results, './Kernel_results.txt')
                if self.evaluation == mean_squared_error:
                    hyper_tuning_curve_3D(hyper_space, np.array(self.training_errors), np.array(self.test_errors), method='Error')
                elif self.evaluation == r2_score:
                    hyper_tuning_curve_3D(hyper_space, np.array(self.training_errors), np.array(self.test_errors), method='Accuracy')
            if isinstance(model, DCKRR):
                unique_gammas = np.unique(hyper_space[:, 1], return_index=False)
                unique_alphas = np.unique(hyper_space[:,0], return_index=False)
                results = pd.DataFrame()
                results["Gamma"] = unique_gammas
                results["Alpha"] = unique_alphas
                if self.evaluation == mean_squared_error:
                    results["Training_errors"] = np.array(self.training_errors)
                    results["Test_errors"] = np.array(self.test_errors)
                    results["Exp. training error estimate"] = np.min(np.array(self.training_errors))
                    results["Risk estimate (Exp. stat. risk estimate)"] = np.min(np.array(self.test_errors))
                if self.evaluation == r2_score:
                    results["Training_accuracy"] = np.array(self.training_errors)
                    results["Test_accuracy"] = np.array(self.test_errors)
                    results["Exp. training acc. estimate"] = np.max(np.array(self.training_errors))
                    results["Accuracy estimate (Exp. stat. acc. estimate)"] = np.max(np.array(self.test_errors))
                print(results)
                results.to_csv('./Kernel_results.csv', index=False)
                df_to_latex_table(results, './Kernel_results.txt')
                
        
        return self.mean_training_error, self.risk_esimate, self.hyper_opt, self.predictors, m
    
    

    def learningCurveCV(self, X, y, hyper_space, type, samples):
        """
        Computes the learning curve using a specific cross-validation method.
        
        Parameters:
        - X: Feature matrix
        - y: Target variable
        - hyper_space: Hyperparameters space
        - type: Type of cross-validation ("nestedKFoldCV" or "bestCV")
        - samples: list of dataset sizes to consider to build the learning curve

        """
        Xy = pd.concat([X, y], axis=1)
        Xy_part = [Xy.iloc[:i] for i in range(Xy.shape[0]//samples, Xy.shape[0] + 1, Xy.shape[0]//samples)]
        features = self.features
        target = self.target
        for i, Xy_sub in enumerate(Xy_part):
            if self.verbose : 
                print("\n###############################Estimating the risk for "+str(i)+"-th sample###################################")
            X = Xy_sub[features]
            y = Xy_sub[target]
            if type=='nestedKFoldCV':
                train, test, hyper, predict, m = self.nestedKFoldCV(X, y, hyper_space)
            elif type=='bestCV':
                train, test, hyper, predict, m = self.bestCV(X, y, hyper_space)
            self.learningC_train.append(train)
            self.learningC_test.append(test)
            self.learningC_hyper.append(hyper)
            self.learningC_model.append(predict)
            self.learningC_ms.append(m)

        if self.evaluation == mean_squared_error:
            plot_learning_curve(self.learningC_ms, self.learningC_train, self.learningC_test, method='Error')
        elif self.evaluation == r2_score:
            plot_learning_curve(self.learningC_ms, self.learningC_train, self.learningC_test, method='Accuracy')
        plot_hyperparam_curve(self.learningC_ms, list(zip(*self.learningC_hyper)))


    

    def save(self, path):
        """
        Saves the risk estimation results to a specified path (nestedKFoldCV version).
        
        Parameters:
        - path: Path where results should be saved

        """
        results = pd.DataFrame()
        results["Fold"] = ["Fold n."+str(i) for i in range(self.K)]
        results["Hyperparameter"] = self.hyper_opt
        if self.evaluation == mean_squared_error:
            results["Training_error"] = self.training_errors
            results["Test_error"] = self.test_errors
            results["Exp. training error estimate"] = self.mean_training_error
            results["Risk estimate (Exp. stat. risk estimate)"] = self.risk_esimate
        if self.evaluation == r2_score:
            results["Training_accuracy"] = self.training_errors
            results["Test_accuracy"] = self.test_errors
            results["Exp. training acc. estimate"] = self.mean_training_error
            results["Accuracy estimate (Exp. stat. acc. estimate)"] = self.risk_esimate
        print(results)
        results.to_csv(path+'/risk_estimate_nestedKFoldCV.csv', index=False)
        df_to_latex_table(results, path+'/risk_estimate_nestedKFoldCV.txt')

    
    def save_best(self, path):
        """
        Saves the risk estimation results to a specified path (bestCV version).
        
        Parameters:
        - path: Path where results should be saved
        
        """
        column_2 = "Exp. training error estimate" if self.evaluation == mean_squared_error else "Exp. training acc. estimate"
        column_3 = "Risk estimate" if self.evaluation == mean_squared_error else "Accuracy estimate"
        if type(self.hyper_opt) is not list:
            results = pd.DataFrame({
                "Hyperparameter 1": self.hyper_opt[0],
                "Hyperparameter 2": self.hyper_opt[1],
                column_2: self.mean_training_error,
                column_3: self.risk_esimate,
            }, index=[0])
        else: 
            results = pd.DataFrame({
                "Hyperparameter": self.hyper_opt,
                column_2: self.mean_training_error,
                column_3: self.risk_esimate,
            }, index=[0])
        print(results)
        results.to_csv(path+'/risk_estimate_bestCV.csv', index=False)
        df_to_latex_table(results, path+'/risk_estimate_bestCV.txt')
    
    
    def save_predictors(self, path):
        """
        Saves the predictors for each fold to a specified path.

        Parameters:
        - path: Path where predictors should be saved

        """
        pred = np.array(self.predictors)
        colos = ["w_"+str(i) for i in range(pred.shape[1])]
        pred_dataframe = pd.DataFrame(pred, columns=colos)
        pred_dataframe.insert(0, "Fold", ["Fold n."+str(i) for i in range(self.K)])
        print(pred_dataframe)
        pred_dataframe.to_csv(path+'/risk_estimate_predictor.csv', index=False)
        df_to_latex_table(pred_dataframe, path+'/risk_estimate_predictor.txt')


    def get_risk_estimate(self):
        return self.risk_esimate
    
    def get_test_errors(self):
        return self.test_errors
    
    def get_hyper_opt(self):
        return self.hyper_opt
    
    def get_predictors(self):
        return self.predictors

    def set_visualize(self, visualize):
        self.visualize = visualize
    
    def set_verbose(self, verbose):
        self.verbose = verbose
