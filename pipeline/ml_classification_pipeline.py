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
from sklearn.model_selection import StratifiedKFold



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

            # Train model on processed train set
            self.model_handler.train(X_train_processed, y_fold_train)
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
        model_params = self.model_handler.model.get_params()

        # Convert DataFrame to dictionary
        predictions_dict = predictions_df.to_dict(orient='records')

        # Construct JSON object
        json_object = {f'{model_name}_{str(model_params)}': predictions_dict}

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
