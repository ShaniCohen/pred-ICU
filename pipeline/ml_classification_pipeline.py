from data_handler import DataHandler
from preprocessing import Preprocessing
from model_handler import ModelHandler
import logging
from scipy.stats import norm
import numpy as np
from sklearn.metrics import classification_report
# from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import datetime
import os
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
import optuna
import shap
import pickle
import xgboost as xgb



class MLClassificationPipeline:
    def __init__(self, data_handler: DataHandler, preprocessing: Preprocessing, model_handler: ModelHandler,impute, number_of_splits, do_shap, to_scale, to_optimize_hyperparams, do_cross_val):
        self.data_handler = data_handler
        self.preprocessing = preprocessing
        self.model_handler = model_handler
        self.splits_for_cv = number_of_splits
        self.impute=impute
        self.do_shap = do_shap
        self.to_scale = to_scale
        self.to_optimize_hyperparams = to_optimize_hyperparams
        self.do_cross_val = do_cross_val
        # self.data_handler = data_handler
        # self.preprocessing = preprocessing

    # def run_pipeline(self):
    #     # Load and split data
    #     main_df = self.data_handler.load_data()
    #     logging.info(f'finished loading data')
    #     logging.info(f'main_df shape: {main_df.shape}')
    #
    #     # print('main_df shape:', main_df.shape)
    #     X_train, X_test, y_train, y_test = self.data_handler.split_data(main_df)
    #     logging.info(f'X_train shape: {X_train.shape}')
    #     logging.info(f'X_test shape: {X_test.shape}')
    #     logging.info(f'y_train shape: {y_train.shape}')
    #     logging.info(f'y_test shape: {y_test.shape}')
    #
    #     # Fit preprocessing steps on the train set
    #     # get a list of the x features for the model
    #     x_features_list = [col for col in X_train.columns if col != 'hospital_death']
    #     X_train_processed, fitted_scaler, feature_info_dtype, dict_of_fill_values, encoder_info = self.preprocessing.run_preprocessing_fit(data=X_train, list_of_x_features_for_model=x_features_list)
    #     logging.info(f'finished preprocessing fit')
    #     logging.info(f'feature_info_dtype: {feature_info_dtype}')
    #     logging.info(f'dict_of_fill_values: {dict_of_fill_values}')
    #
    #     # # Transform both train and test sets
    #     # X_train_processed = self.preprocessing.run_preprocessing_transform(
    #     #     data=X_train,
    #     #     scaler=fited_scaler,
    #     #     feature_info_dtype=feature_info_dtype,
    #     #     dict_of_fill_values=dict_of_fill_values,
    #     #     encoder_information=encoder_info
    #     # )
    #     X_test_processed = self.preprocessing.run_preprocessing_transform(
    #         data=X_test,
    #         scaler=fitted_scaler,
    #         feature_info_dtype=feature_info_dtype,
    #         dict_of_fill_values=dict_of_fill_values,
    #         encoder_information=encoder_info
    #     )
    #     logging.info(f'finished preprocessing transform')
    #     logging.info('X_train_processed shape after preprocessing:', X_train_processed.shape)
    #     logging.info('X_test_processed shape after preprocessing:', X_test_processed.shape)
    #
    #     # Train model on processed train set
    #     self.model_handler.train(X_train_processed, y_train)
    #     logging.info(f'finished training')
    #
    #     # Predict and evaluate on processed test set
    #     predictions = self.model_handler.predict(X_test_processed)
    #     logging.info(f'finished predicting')
    #
    #     # Add evaluation metrics here
    #     # Calculate precision, recall, f1-score, and support
    #     # print(f'classification_report \n{classification_report(y_test, predictions)}')
    #     logging.info(f'classification_report: \n{classification_report(y_test, predictions)}')
    #     # print(f'precision_score \n{precision_score(y_test, predictions)}')
    #     logging.info(f'precision_score: {precision_score(y_test, predictions)}')
    #     # print(f'recall_score \n{recall_score(y_test, predictions)}')
    #     logging.info(f'recall_score: {recall_score(y_test, predictions)}')
    #     # print(f'f1_score \n{f1_score(y_test, predictions)}')
    #     logging.info(f'f1_score: {f1_score(y_test, predictions)}')
    #
    #     # calculate auroc
    #     auroc = roc_auc_score(y_test, predictions)
    #     logging.info(f'calculate AUROC of the model: {auroc}')
    #
    #     # plot the roc curve
    #     fpr, tpr, thresholds = roc_curve(y_test, predictions)
    #     plt.plot([0, 1], [0, 1], linestyle='--')
    #     plt.plot(fpr, tpr, marker='.', label='AUROC = %0.3f ()' % auroc)
    #     plt.xlabel('1- Specificity')
    #     plt.ylabel('Sensitivity')
    #     plt.legend(handlelength=0)
    #     plt.show()

    # def get_model_params(self,model_name):
    #     # XGBoost parameters
    #     xgboost_params = {
    #         'learning_rate': Real(0.01, 0.5, prior='uniform'),
    #         'n_estimators': Integer(100, 1000),
    #         'max_depth': Integer(3, 9),
    #         'subsample': Real(0.8, 1.0, prior='uniform'),
    #         'colsample_bytree': Real(0.8, 1.0, prior='uniform'),
    #     }

    #     # Parameter space for Logistic Regression
    #     logistic_params = {
    #         'C': Real(0.01, 100.0, prior='log-uniform'),
    #         'penalty': Categorical(['l1', 'l2']),
    #         'solver': Categorical(['liblinear', 'saga'])
    #     }

    #     # Parameter space for Random Forest
    #     random_forest_params = {
    #         'n_estimators': Integer(10, 1000),
    #         'max_depth': Integer(1, 50),
    #         'min_samples_split': Integer(2, 100),
    #         'min_samples_leaf': Integer(1, 50),
    #         'max_features': Categorical(['sqrt', 'log2', None])
    #     }
    #     # xgboost_params = {
    #     #     # 'learning_rate': [0.1, 0.5],
    #     #     'n_estimators': [100,500],
    #     #     # 'max_depth': [ 5, 7, 9],
    #     #     # 'subsample': [0.8, 0.9, 1.0],
    #     #     # 'colsample_bytree': [0.8, 0.9, 1.0],
    #     # }

    #     # # RandomForest parameters
    #     # random_forest_params = {
    #     #     'n_estimators': [200,500],
    #     #     # 'max_depth': [None, 10, 20, 30],
    #     #     # 'min_samples_split': [2, 5, 10],
    #     #     # 'min_samples_leaf': [1, 2, 4],
    #     # }

    #     # # Logistic Regression parameters
    #     # logistic_params = {
    #     #     # 'penalty': ['l1', 'l2'],
    #     #     'C': [0.01, 0.1, 1, 10],
    #     #     #'solver': ['liblinear', 'saga'],
    #     # }
    #     # print(f'model_name: {model_name}')
    #     if model_name == 'LogisticRegression':
    #         return logistic_params
    #     if model_name == 'RandomForestClassifier':
    #         return random_forest_params
    #     if model_name == 'XGBClassifier':
    #         return xgboost_params
        
    
    
    # def tune_hyperparameters_Bayesian(self, model, search_spaces, X_train, y_train, cv):
    #     """
    #     Perform Bayesian optimization to find the best hyperparameters.
    #     """
    #     bayes_search = BayesSearchCV(
    #         estimator=model,
    #         search_spaces=search_spaces,
    #         n_iter=5,  # Adjust the number of iterations
    #         scoring='f1',
    #         cv=cv,
    #         verbose=1,
    #         n_jobs=-1
    #     )
    #     bayes_search.fit(X_train, y_train)
    #     best_params = bayes_search.best_params_
    #     best_score = bayes_search.best_score_
    #     return best_params, best_score


    # def tune_hyperparameters(self,model, param_grid, X_train, y_train, cv):
    #     """
    #     Perform grid search cross-validation to find the best hyperparameters.
    #     """
    #     grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=cv, verbose=1, n_jobs=-1)
    #     grid_search.fit(X_train, y_train)
    #     best_params = grid_search.best_params_
    #     best_score = grid_search.best_score_
    #     return best_params,best_score
# ------------------------------------------------------------------------------------------
# ----------------------------------- SHAP --------------------------------------------------

    def plot_shap_waterfall(base_value, shap_values, feature_names, actual_prediction):
        """
        Plots a SHAP waterfall chart for a single prediction.
        
        Parameters:
        - base_value: The base value (average model output over the dataset).
        - shap_values: Array of SHAP values for each feature for a single prediction.
        - feature_names: List of feature names corresponding to the SHAP values.
        - actual_prediction: The actual prediction for the instance.
        """
        # Start with the base value
        start_value = base_value
        # Initialize the cumulative sum of SHAP values
        cum_shap_values = [start_value]
        # Calculate cumulative sum
        for shap_value in shap_values:
            start_value += shap_value
            cum_shap_values.append(start_value)
        
        # Prepare the plotting data
        step_values = [base_value] + list(cum_shap_values[:-1])
        end_values = cum_shap_values
        
        # Plotting
        plt.figure(figsize=(10, 6))
        for i in range(len(shap_values)):
            plt.fill_between([i, i + 1], step_values[i], end_values[i], color='skyblue' if shap_values[i] >= 0 else 'salmon', step='pre')
        
        # Adding final prediction
        plt.plot([0, len(shap_values)], [actual_prediction, actual_prediction], 'k--', label='Final Prediction')
        
        # Customize the plot
        plt.xticks(ticks=range(len(feature_names) + 1), labels=['Base Value'] + feature_names, rotation=45, ha="right")
        plt.ylabel('Prediction Value')
        plt.title('SHAP Waterfall Plot for a Single Prediction')
        plt.legend()
        plt.grid(True)
        
        # Show plot
        plt.tight_layout()
        plt.show()

    # Example usage (assuming you have the necessary values calculated)
    # base_value = 0.5  # Example base value
    # shap_values = [0.1, -0.2, 0.05, 0.3]  # Example SHAP values for each feature for a single prediction
    # feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4']  # Example feature names
    # actual_prediction = 0.75  # Example actual prediction for the instance

    # plot_shap_waterfall(base_value, shap_values, feature_names, actual_prediction)
    # # Plot the SHAP waterfall chart
    
    
    # fit the model for calculating shap values
    # because we need to keep the interpretability of the model we will not be using scaling
    # also xgboost is a good model because it does not require scaling
    # def fit_model_for_shap(self):
    #     # before we do all the preprocessing, we will use the class of the preprocessing for that
    #     # use Data handler class to load the data
    #     main_data = self.data_handler.load_data()
        
    #     X = main_data.drop(['encounter_id', 'patient_id', 'hospital_death', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'readmission_status'], axis=1)
    #     y = main_data['hospital_death']
    #     x_features_list = X.columns.tolist()
    #     X_train_processed, fitted_scaler, feature_info_dtype, dict_of_fill_values, encoder_info = self.preprocessing.run_preprocessing_fit(data=X, list_of_x_features_for_model=x_features_list, to_scale=False)
    #     model = self.model_shap
    #     model.fit(X_train_processed, y)
    #     return model, X_train_processed, y

    def calculate_and_save_shap_values(self, model, X):
        """
        Calculates SHAP values for a given model and dataset, then saves the values to a pickle file.
        
        Parameters:
        - model: A trained machine learning model.
        - X: The dataset (features) for which SHAP values are to be calculated (numpy array or pandas DataFrame).
        - path: The file path where the SHAP values should be saved (string).
        """
        # Initialize the SHAP Tree explainer (use KernelExplainer, DeepExplainer, or GradientExplainer as needed for other model types)
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # # Directory path for JSON file
        # base_directory = os.path.dirname(os.path.dirname(DataHandler.file_path))
        # shap_directory = os.path.join(base_directory, 'shap_values')
        # if not os.path.exists(shap_directory):
        #     os.makedirs(shap_directory)

        # # File path for JSON file
        # date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # json_file_path = os.path.join(shap_directory, f'shap_values_{date}.pkl')
        
        # # Save the SHAP values to a pickle file
        # with open(json_file_path, 'wb') as file:
        #     pickle.dump(shap_values, file)

        # # save the explainer
        # explainer_file_path = os.path.join(shap_directory, f'explainer_{date}.pkl')
        # with open(explainer_file_path, 'wb') as file:
        #     pickle.dump(explainer, file)

        return shap_values, explainer

    # def run_shap(self):
    #     # Fit the model
    #     model, X_train_processed, _ = self.fit_model_for_shap()
    #     # Calculate and save SHAP values
    #     self.calculate_and_save_shap_values(model, X_train_processed)
    #     # print(f"SHAP values saved to {path}")



    def tune_hyperparameters_Optuna(self, model, X_train, y_train, cv):
        def objective(trial):
            model_name = type(model).__name__
            
            if model_name == 'XGBClassifier':
                param = {
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 9),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                }
            elif model_name == 'RandomForestClassifier':
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
            elif model_name == 'LogisticRegression':
                param = {
                    'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                }
            
            model.set_params(**param)
            # Initialize StratifiedKFold
            

            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)  # Modify n_trials as needed

        best_params = study.best_params
        best_score = study.best_value

        return best_params, best_score

        
    # Yuval's version (probabilities instead of binary predictions):
    def run_pipeline(self):
        # Load and split data
        main_df = self.data_handler.load_data()
        logging.info(f'finished loading data')
        logging.info(f'main_df shape: {main_df.shape}')



        # print('main_df shape:', main_df.shape)
        # in we want to use the split data
        # X_train, X_test, y_train, y_test = self.data_handler.split_data(main_df)
        
        # in case we want to use the entire data set
        X = main_df.drop(['encounter_id', 'patient_id', 'hospital_death', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'readmission_status'], axis=1)
        y = main_df['hospital_death']

        ## edit fold- gal
        #creating new feauteres 
        X["age_square"]=X["age"]**2
        X["age_power_three"]=X["age"]**3
        #drop BMI ( fill it later)
        X.drop(columns='bmi',inplace=True)

        cv_strategy = StratifiedKFold(n_splits=self.splits_for_cv)
        # results = {'folds': []}
        # Initialize lists to store aggregated results
        all_y_test = []
        all_probabilities = []
        all_binary_predictions = []
        all_patient_id=[]
        best_params_dict = {}
        best_score_dict = {}
        results_shap = {}
        if self.do_cross_val:
            for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y)):
                X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
                y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]


                logging.info(f'X_train shape: {X_fold_train.shape}')
                logging.info(f'X_test shape: {X_fold_test.shape}')
                logging.info(f'y_train shape: {y_fold_train.shape}')
                logging.info(f'y_test shape: {y_fold_test.shape}')

                #Galfold- add imputation
                X_fold_train, X_fold_test,y_fold_train, y_fold_test=self.impute.execute_imputation(X_fold_train, X_fold_test,y_fold_train, y_fold_test)
            
                # Fit preprocessing steps on the train set
                # get a list of the x features for the model
                x_features_list = [col for col in X_fold_train.columns if col != 'hospital_death']
                X_train_processed, fitted_scaler, feature_info_dtype, dict_of_fill_values, encoder_info = self.preprocessing.run_preprocessing_fit(data=X_fold_train, 
                                                                                                                                                list_of_x_features_for_model=x_features_list,
                                                                                                                                                to_scale=self.to_scale)
                logging.info(f'finished preprocessing fit')
                logging.info(f'feature_info_dtype: {feature_info_dtype}')
                logging.info(f'dict_of_fill_values: {dict_of_fill_values}')

                X_test_processed = self.preprocessing.run_preprocessing_transform(
                    data=X_fold_test,
                    scaler=fitted_scaler,
                    feature_info_dtype=feature_info_dtype,
                    dict_of_fill_values=dict_of_fill_values,
                    encoder_information=encoder_info,
                    to_scale=self.to_scale
                )
                logging.info(f'finished preprocessing transform')
                logging.info(f'X_train_processed shape after preprocessing: {X_train_processed.shape}')
                logging.info(f'X_test_processed shape after preprocessing: {X_test_processed.shape}')

                # Get model name and parameters
                model_name = self.model_handler.model.__class__.__name__
                # param_grid = self.get_model_params(model_name)
                
                # # Tune hyperparameters for the current fold
                # best_params,best_score = self.tune_hyperparameters_Bayesian(model=self.model_handler.model, 
                #                                                             search_spaces=param_grid, 
                #                                                             X_train=X_train_processed, 
                #                                                             y_train=y_fold_train, 
                #                                                             cv=3)

                # Call the tuning function
                if self.to_optimize_hyperparams:
                    print(f'starting optimization for fold {fold}')
                    stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    best_params, best_score = self.tune_hyperparameters_Optuna(model=self.model_handler.model, X_train=X_train_processed, y_train=y_fold_train, cv=stratified_cv)

                    print("Best parameters:", best_params)
                    print("Best score:", best_score)

                    # best_params,best_score = self.tune_hyperparameters(self.model_handler.model, 
                    #                                                    param_grid, 
                    #                                                    X_train_processed, 
                    #                                                    y_fold_train, 
                    #                                                    3)
                    print(f'model name: {model_name} best params: {best_params}')
                    best_params_dict[fold] = best_params
                    best_score_dict[fold] = best_score
                    self.model_handler.model.set_params(**best_params)

                if model_name == 'XGBClassifier':
                    eval_set = [(X_test_processed, y_fold_test)]
                    self.model_handler.model.fit(X_train_processed, y_fold_train, eval_set=eval_set)
                    # self.model_handler.model.fit(X_train_processed, y_fold_train, eval_metric='logloss')
                    # self.model_handler.train(X_train_processed, y_fold_train)
                else:
                    self.model_handler.train(X_train_processed, y_fold_train)
                # Train model on processed train set
                
                logging.info('finished training')

                if self.do_shap:
                    if fold == self.splits_for_cv -1 : #| (fold == self.splits_for_cv -2)
                        # --------------------------------------------------------
                        
                
                        # Fit a XGBClassifier model
                        seed = 1
                        reg_alpha_param = 0.2
                        np.random.seed(42)
                        model = xgb.XGBClassifier(random_state=seed, 
                                                alpha=reg_alpha_param,
                                                eval_metric='logloss', 
                                                early_stopping_rounds=10,verbose=0,verbose_eval=False)
                        eval_set = [(X_test_processed, y_fold_test)]
                        trained_model = model.fit(X_train_processed, y_fold_train, eval_set=eval_set)
                        
                        
                        # -----------------------------------------------------------
                        # create shap values
                        xgb_explainer = shap.TreeExplainer(trained_model)
                        # Calculate SHAP values
                        xgb_shap_values = xgb_explainer(X_train_processed)
                        xgb_feature_names = xgb_shap_values.feature_names
                        # ----------------------------------------------------------

                        # make ICE plots - Individual Conditional Expectation
                        # Step 2: Get the probabilities of the positive class for each instance
                        probs = trained_model.predict_proba(X_train_processed)[:, 1]  # Assuming we're interested in the second class

                        # Define the probability ranges
                        prob_ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), 
                                    (0.4, 0.5), (0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1)]
                        list_columns_for_ice = ['age','gcs_motor_apache','gcs_verbal_apache','d1_bun_min','d1_spo2_min']
                        for ice_column in list_columns_for_ice:
                            samples = pd.DataFrame()
                            list_of_indices = []
                            for lower, upper in prob_ranges:
                                # Find indices where probability falls into the current range
                                range_indices = np.where((probs >= lower) & (probs < upper))[0]
                                
                                # Check if there are any indices in this range
                                if range_indices.size > 0:
                                    # Randomly select one index from this range
                                    selected_index = np.random.choice(range_indices)
                                    list_of_indices.append(selected_index)
                                    # Get the corresponding instance from X_train_processed
                            samples = X_train_processed.iloc[list_of_indices]

                            # --------------------------------------------------
                            
                            # Step 4: Create a DataFrame with the instances
                            df_samples = pd.DataFrame(samples, columns=samples.columns)

                            # Step 5: Generate a range of values for the 'age' feature
                            column_range = np.linspace(X_train_processed[f'{ice_column}'].min(), X_train_processed[f'{ice_column}'].max(), 100).astype(int)
                            df_full = pd.DataFrame(index=np.arange(len(df_samples) * len(column_range)), columns=df_samples.columns)
                            
                            
                            # Assuming 'samples' is a DataFrame and not a 2D numpy array
                            # Assign each sample to each age in the range
                            for i, idx in enumerate(list_of_indices):
                                sample_row = samples.iloc[i]  # Use iloc to select the row by integer location
                                for j, age in enumerate(column_range):
                                    df_full_index = i * len(column_range) + j
                                    df_full.iloc[df_full_index] = sample_row.to_dict()
                                    df_full.at[df_full_index, f'{ice_column}'] = age
                            
                            # Step 1: Explicitly convert data types to match those in X_train_processed
                            for column in df_full.columns:
                                df_full[column] = df_full[column].astype(X_train_processed[column].dtype)

                            # print(f'df_full shape: {df_full.shape}')
                            # print(f'df_full columns: {df_full.columns}')
                            # print(f'df_full dtypes: {df_full.dtypes}')

                            # Step 6: Make new predictions on the new DataFrame using the trained model
                            if 'probability' in df_full.columns:
                                df_full = df_full.drop('probability', axis=1)
                            df_full['probability'] = trained_model.predict_proba(df_full)[:, 1]

                            # Step 7: Plot the new ICE plot using the new DataFrame
                            for i, idx in enumerate(list_of_indices):
                                plt.plot(column_range, df_full.iloc[i*len(column_range):(i+1)*len(column_range)]['probability'], label=f'Instance {i+1}')

                            plt.title('ICE - Individual Conditional Expectation')
                            plt.xlabel(f'{ice_column}')
                            plt.ylabel('Probability')
                            # plt.legend()
                            plt.show()
                        # --------------------------------------------------------
                        # # --------------------------------------------------------
                        # calculate shap values and save in data frame
                        
                        # create csv file for shap values, the columns will be the features names, the shap values, the normalized shap values (shap value/sum(shap values))
                        # also sort the data frame by the shap values
                        list_columns = ['feature_name', 'shap_value', 'normalized_shap_value']
                        shap_values_df = pd.DataFrame(columns=list_columns)
                        list_shape_values = []
                        list_shap_nurmalized_values = []
                        list_feature_names = []
                        for name, shap_val in zip(xgb_feature_names, xgb_shap_values.values[0]):
                            list_feature_names.append(name)
                            list_shape_values.append(shap_val)
                            list_shap_nurmalized_values.append(shap_val/sum(xgb_shap_values.values[0]))
                        shap_values_df['feature_name'] = list_feature_names
                        shap_values_df['shap_value'] = list_shape_values
                        shap_values_df['normalized_shap_value'] = list_shap_nurmalized_values
                        shap_values_df = shap_values_df.sort_values(by='shap_value', ascending=False)
                        # save the shap values to a csv file
                        # Directory path for JSON file
                        base_directory = os.path.dirname(os.path.dirname(self.data_handler.file_path))
                        # print(f'base_directory: {base_directory}')
                        shap_directory = os.path.join(base_directory, 'shap_csv_files')
                        # print(f'shap_directory: {shap_directory}')
                        if not os.path.exists(shap_directory):
                            os.makedirs(shap_directory)
                        else:
                            print(f'shap_directory already exists')

                        # File path for JSON file
                        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        shap_file_path = os.path.join(shap_directory, f'shap_values_{date}.csv')
                        shap_values_df.to_csv(shap_file_path, index=False)



                    
                        number_of_shap_features = 20
                        print(f'starting SHAP for fold {fold}')
                        # Calculate and save SHAP values
                        seed = 1
                        reg_alpha_param = 0.2
                        np.random.seed(42)
                        model = xgb.XGBClassifier(random_state=seed, 
                                                alpha=reg_alpha_param,
                                                eval_metric='logloss', 
                                                early_stopping_rounds=10,verbose=0,verbose_eval=False)
                        eval_set = [(X_test_processed, y_fold_test)]
                        trained_model = model.fit(X_train_processed, y_fold_train, eval_set=eval_set)
                        # save the trained model in a pikle file
                        # Directory path for JSON file
                        base_directory = os.path.dirname(os.path.dirname(self.data_handler.file_path))
                        shap_model_directory = os.path.join(base_directory, 'shap_model')
                        with open(shap_model_directory, 'wb') as file:
                            pickle.dump(trained_model, file)

                        # --------------------------------------------------------

                        # Define the importance types and their descriptions for the x-axis
                        importance_types = {
                            'gain': 'Average Gain',
                            'weight': 'Feature Frequency (Count)',
                            'cover': 'Average Coverage'
                        }

                        for importance_type, description in importance_types.items():
                            # Assuming trained_model is your XGBoost model
                            # Fetch feature importance scores
                            scores = trained_model.get_booster().get_score(importance_type=importance_type)
                            # Create a Series for the scores
                            importance_series = pd.Series(scores).sort_values(ascending=False)
                            
                            # If there are more than number_of_shap_features features, select the top number_of_shap_features
                            if len(importance_series) > number_of_shap_features:
                                importance_series = importance_series.head(number_of_shap_features)
                            
                            # Reverse the series for plotting
                            importance_series = importance_series.iloc[::-1]
                            
                            # Use different colors for each bar
                            colors = plt.cm.viridis(np.linspace(0, 1, len(importance_series)))
                            
                            # Plot
                            plt.figure(figsize=(10, 8))
                            importance_series.plot(kind='barh', color=colors, title=f'XGBoost Feature Importance ({description})')
                            plt.xlabel(description)
                            plt.ylabel('Features')
                            plt.show()


                        # --------------------------------------------------------

                        
                        # plot waterfall plot for 4 types of cases or persons          
                        predict_proba = trained_model.predict_proba(X_train_processed)[:,1]
                        shap_joined_df = X_train_processed.copy()
                        shap_joined_df['hospital_death'] = y_fold_train
                        low_threshold = np.percentile(predict_proba,10)
                        high_threshold = np.percentile(predict_proba,90)
                        shap_joined_df['predict_proba'] = predict_proba

                        # get the indexs of each case
                        # case 1 - a person died ('hospital_death'==1  & the shap model says a low probability for dying)
                        # case 2 - a person died ('hospital_death'==1  & the shap model says a high probability for dying)
                        # case 3 - a person didn't died ('hospital_death'==0  & the shap model says a low probability for dying)
                        # case 4 - a person didn't died ('hospital_death'==0  & the shap model says a High probability for dying)
                        conditions = [((shap_joined_df['hospital_death'] == 1) & (shap_joined_df['predict_proba'] < low_threshold)), 
                                    ((shap_joined_df['hospital_death'] == 1) & (shap_joined_df['predict_proba'] > high_threshold)), 
                                    ((shap_joined_df['hospital_death'] == 0) & (shap_joined_df['predict_proba'] < low_threshold)), 
                                    ((shap_joined_df['hospital_death'] == 0) & (shap_joined_df['predict_proba'] > high_threshold))]

                        labels = ['case 1', 'case 2', 'case 3', 'case 4']
                        shap_joined_df['case'] = np.select(conditions, labels, default='other')
                        # get the first index for each case
                        case_1_index = shap_joined_df[shap_joined_df['case'] == 'case 1'].index[0]
                        case_2_index = shap_joined_df[shap_joined_df['case'] == 'case 2'].index[0]
                        case_3_index = shap_joined_df[shap_joined_df['case'] == 'case 3'].index[0]
                        case_4_index = shap_joined_df[shap_joined_df['case'] == 'case 4'].index[0]

                        # get the probability for each case
                        case_1_probability = shap_joined_df.loc[case_1_index,'predict_proba']
                        case_2_probability = shap_joined_df.loc[case_2_index,'predict_proba']
                        case_3_probability = shap_joined_df.loc[case_3_index,'predict_proba']
                        case_4_probability = shap_joined_df.loc[case_4_index,'predict_proba']
                        # print all the probabilities 
                        # print(f'case 1 probability: {case_1_probability}')
                        # print(f'case 2 probability: {case_2_probability}')
                        # print(f'case 3 probability: {case_3_probability}')
                        # print(f'case 4 probability: {case_4_probability}')


                        # print(f'case 1 index: {case_1_index}, with pobability of:{}')
                        # print(f'case 2 index: {case_2_index}')
                        # print(f'case 3 index: {case_3_index}')
                        # print(f'case 4 index: {case_4_index}')

                        # get the shap values for each case
                        shap_values_case_1 = xgb_shap_values[case_1_index]
                        shap_values_case_2 = xgb_shap_values[case_2_index]
                        shap_values_case_3 = xgb_shap_values[case_3_index]
                        shap_values_case_4 = xgb_shap_values[case_4_index]

                        # plot the waterfall plot for each case and add the title of the case
                        # case 1
                        print(f'case 1 - a person died & the shap model says a low probability for dying')
                        shap.plots.waterfall(shap_values_case_1, max_display=number_of_shap_features)  
                        # case 2
                        print(f'case 2 - a person died & the shap model says a high probability for dying')
                        shap.plots.waterfall(shap_values_case_2, max_display=number_of_shap_features)
                        # case 3
                        print(f'case 3 - a person didn\'t died & the shap model says a low probability for dying')
                        shap.plots.waterfall(shap_values_case_3, max_display=number_of_shap_features)
                        # case 4
                        print(f'case 4 - a person didn\'t died & the shap model says a High probability for dying')
                        shap.plots.waterfall(shap_values_case_4, max_display=number_of_shap_features)
                    
                        # some summerize plots   
                        fig = plt.figure()             
                        shap.summary_plot(xgb_shap_values, 
                                        X_train_processed, 
                                        feature_names=xgb_feature_names, 
                                        plot_type="bar", 
                                        max_display=number_of_shap_features,
                                        show=False)
                        plt.xlabel('Mean |SHAP Value|')
                        plt.show()
                        shap.summary_plot(xgb_shap_values, X_train_processed, feature_names=xgb_feature_names, max_display=number_of_shap_features)
                        # #  --------------------------------------------------------
                        # # Plot dependence plots for each of the top k features
                        # k=8
                        # # Step 1: Summarize the absolute SHAP values across all samples to get feature importance
                        # shap_sum = np.abs(xgb_shap_values.values).mean(axis=0)
                        
                        # # For models with multi-class outputs, adjust the dimension used for mean calculation
                        # if len(shap_sum.shape) > 1:
                        #     shap_sum = shap_sum.mean(axis=0)
                        
                        # # Identify the top k features based on the average magnitude of SHAP values
                        # top_indices = np.argsort(shap_sum)[-k:]
                        # # feature_names = xgb_shap_values.feature_names

                        # # Step 3: Plot dependence plots for each of the top k features
                        # for index in top_indices:
                        #     shap.dependence_plot(index, xgb_shap_values.values, X_train_processed, feature_names=xgb_feature_names)
                        # --------------------------------------------------------
                        # import numpy as np
                        # import pandas as pd
                        # import matplotlib.pyplot as plt
                        # from sklearn.datasets import make_classification
                        # from xgboost import XGBClassifier

                        # Generate some synthetic classification data
                        # X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
                        # features = ['feature' + str(i) for i in range(X.shape[1])]
                        # X_df = pd.DataFrame(X, columns=features)

                        # # Fit an XGBoost classifier
                        # model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                        # model.fit(X_df, y)

                        # Plot PDP for multiple features (we'll plot for the first two features for example)
                    
                        k = 8
                        # take the best k features from shap values
                        shap_sum = np.abs(xgb_shap_values.values).mean(axis=0)
                        # For models with multi-class outputs, adjust the dimension used for mean calculation
                        if len(shap_sum.shape) > 1:
                            shap_sum = shap_sum.mean(axis=0)
                        # # Identify the top k features based on the average magnitude of SHAP values
                        # top_indices = np.argsort(shap_sum)[-k:]
                        # Ensure top_indices is a 1D array of integers
                        top_indices = np.argsort(shap_sum)[-k:].flatten().astype(int)
                        # print(type(top_indices))
                        # print(top_indices.shape)
                        # print(type(xgb_feature_names))
                        # print(len(xgb_feature_names))
                        features_to_plot = [xgb_feature_names[i] for i in top_indices]
                        n_cols = int(k/2)
                        n_rows = 2

                        fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 10)) # Adjust the figure size as needed
                        ax = ax.ravel() # Flatten the array of axes
                        # features_to_plot = xgb_feature_names[top_indices]
                        # fig, ax = plt.subplots(k, figsize=(10, 5*k))  # Corrected len(k) to k
                        for i, feature in enumerate(features_to_plot):
                            # plot_partial_dependence(trained_model, X_train_processed, [feature], kind='average', target=1, ax=ax[i], grid_resolution=50)
                            PartialDependenceDisplay.from_estimator(trained_model, 
                                                                    X_train_processed, 
                                                                    [feature], 
                                                                    kind='average', 
                                                                    target=1, 
                                                                    ax=ax[i], 
                                                                    grid_resolution=50)
                            
                            # ax[i].set(ylabel="Predicted Probability")
                            ax[i].set_ylabel('Predicted Probability')
                            ax[i].set_title(f'Partial Dependence Plot for {feature}')
                            
                        plt.tight_layout()
                        plt.show()

                    
                        # --------------------------------------------------------


                # Predict on processed test set
                binary_predictions = self.model_handler.predict(X_test_processed)
                probabilities = self.model_handler.predict_proba(X_test_processed)[:, 1]  # Probabilities for the positive class
                logging.info('finished predicting')

                all_patient_id.extend(main_df.loc[test_idx,"patient_id"].values)
                # extand the results to  all lists
                all_y_test.extend(y_fold_test)
                all_probabilities.extend(probabilities)
                all_binary_predictions.extend(binary_predictions)
        else:
            for n in 1000:
                print(f'running for {n} times for bootstrap')
                # get the bootstrap sample
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                # X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
                # y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]


                logging.info(f'X_train shape: {X_fold_train.shape}')
                logging.info(f'X_test shape: {X_fold_test.shape}')
                logging.info(f'y_train shape: {y_fold_train.shape}')
                logging.info(f'y_test shape: {y_fold_test.shape}')

                #Galfold- add imputation
                X_fold_train, X_fold_test,y_fold_train, y_fold_test=self.impute.execute_imputation(X_fold_train, X_fold_test,y_fold_train, y_fold_test)
            
                # Fit preprocessing steps on the train set
                # get a list of the x features for the model
                x_features_list = [col for col in X_fold_train.columns if col != 'hospital_death']
                X_train_processed, fitted_scaler, feature_info_dtype, dict_of_fill_values, encoder_info = self.preprocessing.run_preprocessing_fit(data=X_fold_train, 
                                                                                                                                                list_of_x_features_for_model=x_features_list,
                                                                                                                                                to_scale=self.to_scale)
                logging.info(f'finished preprocessing fit')
                logging.info(f'feature_info_dtype: {feature_info_dtype}')
                logging.info(f'dict_of_fill_values: {dict_of_fill_values}')

                X_test_processed = self.preprocessing.run_preprocessing_transform(
                    data=X_fold_test,
                    scaler=fitted_scaler,
                    feature_info_dtype=feature_info_dtype,
                    dict_of_fill_values=dict_of_fill_values,
                    encoder_information=encoder_info,
                    to_scale=self.to_scale
                )
                logging.info(f'finished preprocessing transform')
                logging.info(f'X_train_processed shape after preprocessing: {X_train_processed.shape}')
                logging.info(f'X_test_processed shape after preprocessing: {X_test_processed.shape}')

                # Get model name and parameters
                model_name = self.model_handler.model.__class__.__name__
                # param_grid = self.get_model_params(model_name)
                

                if model_name == 'XGBClassifier':
                    eval_set = [(X_test_processed, y_fold_test)]
                    self.model_handler.model.fit(X_train_processed, y_fold_train, eval_set=eval_set)
                    # self.model_handler.model.fit(X_train_processed, y_fold_train, eval_metric='logloss')
                    # self.model_handler.train(X_train_processed, y_fold_train)
                else:
                    self.model_handler.train(X_train_processed, y_fold_train)
                # Train model on processed train set
                
                logging.info('finished training')



                # Predict on processed test set
                binary_predictions = self.model_handler.predict(X_test_processed)
                probabilities = self.model_handler.predict_proba(X_test_processed)[:, 1]  # Probabilities for the positive class
                logging.info('finished predicting')

                all_patient_id.extend(main_df.loc[test_idx,"patient_id"].values)
                # extand the results to  all lists
                all_y_test.extend(y_fold_test)
                all_probabilities.extend(probabilities)
                all_binary_predictions.extend(binary_predictions)
            
                
        predictions_df = pd.DataFrame({'y_test':all_y_test,'binary_predictions': all_binary_predictions, 'probabilities': all_probabilities,'patient_id':all_patient_id})
        

        # if self.do_shap:
        #     # Directory path for JSON file
        #     base_directory = os.path.dirname(os.path.dirname(self.data_handler.file_path))
        #     shap_directory = os.path.join(base_directory, 'shap_values')
        #     if not os.path.exists(shap_directory):
        #         os.makedirs(shap_directory)

        #     # File path for JSON file
        #     date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     results_shap_file_path = os.path.join(shap_directory, f'shap_values_{date}.pkl')

        #     # Save JSON object to file
        #     with open(results_shap_file_path, 'wb') as f:
        #         pickle.dump(results_shap, f)


        if self.to_optimize_hyperparams:
            # Get model name and parameters
            model_name = self.model_handler.model.__class__.__name__
            data_frame_best_params = pd.DataFrame(best_params_dict, index=[0])
            data_frame_best_score = pd.DataFrame(best_score_dict, index=[0])
            # name the columns ,fold and params and score
            data_frame_best_params = data_frame_best_params.T.reset_index()
            data_frame_best_params.columns = ['fold', 'params']
            data_frame_best_score = data_frame_best_score.T.reset_index()
            data_frame_best_score.columns = ['fold', 'score']
            
            print (data_frame_best_params)
            print (data_frame_best_score)
            print(f'columns: {data_frame_best_params.columns}')
            print(f'columns: {data_frame_best_score.columns}')
            print(f'dtypes: {data_frame_best_params.dtypes}')
            print(f'dtypes: {data_frame_best_score.dtypes}')
            
            
            # get the best fold by 'score' from the data_frame_best_score
            best_fold = data_frame_best_score.loc[data_frame_best_score['score'].idxmax(),'fold']
            print(best_fold)
        
            # get the best params for the best fold
            best_params = data_frame_best_params.loc[data_frame_best_params['fold'] == best_fold, 
                                                    'params'].values[0]

            print(best_params)

            print(f'model name {model_name} best_params: {best_params}')
        
        else:
            best_params = self.model_handler.model.get_params()
            # print(f'model name {model_name} best_params: {best_params}')

        # Convert DataFrame to dictionary
        predictions_dict = predictions_df.to_dict(orient='records')

        # Construct JSON object
        json_object = {f'{model_name}_{str(best_params)}': predictions_dict}

        # Directory path for JSON file
        base_directory = os.path.dirname(os.path.dirname(self.data_handler.file_path))
        predictions_directory = os.path.join(base_directory, 'predictions')
        if not os.path.exists(predictions_directory):
            os.makedirs(predictions_directory)

        # File path for JSON file
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_file_path = os.path.join(predictions_directory, f'predictions_{date}.json')

        # Save JSON object to file
        with open(json_file_path, 'w') as outfile:
            json.dump(json_object, outfile, indent=4)

        logging.info(f"Predictions saved to {json_file_path}")
        logging.info(f'predictions_df shape: {predictions_df.shape}')
        logging.info(f'predictions_df head: \n{predictions_df.head()}')
        
        return json_file_path
        # # Evaluation Metrics
        # logging.info(f'classification_report: \n{classification_report(y_test, binary_predictions)}')
        # logging.info(f'precision_score: {precision_score(y_test, binary_predictions)}')
        # logging.info(f'recall_score: {recall_score(y_test, binary_predictions)}')
        # logging.info(f'f1_score: {f1_score(y_test, binary_predictions)}')

        # # Calculate AUROC
        # roc_auc, ci_lower, ci_upper = self.compute_auc_confidence_interval(y_test, probabilities)
        # logging.info(f'calculate AUROC of the model: {roc_auc}')
        # fpr, tpr, thresholds = roc_curve(y_test, probabilities)

        # plt.figure(figsize=(10, 10))

        # # Subplot 1 for ROC Curve
        # plt.subplot(1, 2, 1)
        # plt.plot([0, 1], [0, 1], linestyle='--')
        # plt.plot(fpr, tpr, label='AUROC = %0.3f (%0.3f - %0.3f)' % (roc_auc, ci_lower, ci_upper))
        # plt.xlabel('1-Specificity')
        # plt.ylabel('Sensitivity')
        # plt.title('ROC Curve')
        # plt.legend(handlelength=0)

        # # Compute the calibration curve
        # fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probabilities, n_bins=10)

        # # Subplot 2 for Calibration Plot
        # plt.subplot(1, 2, 2)
        # plt.plot(mean_predicted_value, fraction_of_positives, "s-")
        # plt.plot([0, 1], [0, 1], "k:")
        # plt.xlabel('Predicted')
        # plt.ylabel('Observed')
        # plt.title('Calibration Plot')
        # plt.legend(handlelength=0)

        # # Adjust layout
        # plt.tight_layout()
        # plt.show()
