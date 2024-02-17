from data_handler import DataHandler
from preprocessing import Preprocessing
from model_handler import ModelHandler
import logging
from scipy.stats import norm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import datetime
import os
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import optuna


class MLClassificationPipeline:
    def __init__(self, data_handler: DataHandler, preprocessing: Preprocessing, model_handler: ModelHandler,impute, number_of_splits):
        self.data_handler = data_handler
        self.preprocessing = preprocessing
        self.model_handler = model_handler
        self.splits_for_cv = number_of_splits
        self.impute=impute

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
            X_train_processed, fitted_scaler, feature_info_dtype, dict_of_fill_values, encoder_info = self.preprocessing.run_preprocessing_fit(data=X_fold_train, list_of_x_features_for_model=x_features_list,to_scale=True)
            logging.info(f'finished preprocessing fit')
            logging.info(f'feature_info_dtype: {feature_info_dtype}')
            logging.info(f'dict_of_fill_values: {dict_of_fill_values}')

            X_test_processed = self.preprocessing.run_preprocessing_transform(
                data=X_fold_test,
                scaler=fitted_scaler,
                feature_info_dtype=feature_info_dtype,
                dict_of_fill_values=dict_of_fill_values,
                encoder_information=encoder_info,
                to_scale=True
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
                self.model_handler.model.fit(X_train_processed, y_fold_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
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
        
        
        # set the best params to the model
        
        
        # Existing initialization of DataHandler with a file path
        data_handler = DataHandler(file_path=os.path.abspath('..\\data\\training_v2.csv'))

        # # Get the 'data' directory, which is one level up from the file path
        # base_directory = os.path.dirname(os.path.dirname(data_handler.file_path))
        # print(base_directory)

        # # Create a 'predictions' subdirectory inside the 'data' directory
        # predictions_directory = os.path.join(base_directory, 'predictions')
        # if not os.path.exists(predictions_directory):
        #     os.makedirs(predictions_directory)

        # # Use the 'predictions' directory to save the new file
        # date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # file_path = os.path.join(predictions_directory, f'predictions_{date}.csv')
        # predictions_df.to_csv(file_path, index=False)

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
        

        # Convert DataFrame to dictionary
        predictions_dict = predictions_df.to_dict(orient='records')

        # Construct JSON object
        json_object = {f'{model_name}_{str(best_params)}': predictions_dict}

        # Directory path for JSON file
        base_directory = os.path.dirname(os.path.dirname(data_handler.file_path))
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
