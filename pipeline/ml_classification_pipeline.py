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
       


class MLClassificationPipeline:
    def __init__(self, data_handler: DataHandler, preprocessing: Preprocessing, model_handler: ModelHandler):
        self.data_handler = data_handler
        self.preprocessing = preprocessing
        self.model_handler = model_handler
        

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

    def compute_auc_confidence_interval(self, y_test, probabilities, alpha=0.95):
        """
        Compute the confidence interval for the ROC AUC.
        This function uses a normal approximation to the binomial distribution.
        """
        auc = roc_auc_score(y_test, probabilities)
        n1 = sum(y_test)
        n2 = len(y_test) - n1
        q1 = auc / (2 - auc)
        q2 = 2 * auc ** 2 / (1 + auc)
        z = norm.ppf(alpha / 2 + 0.5)
        var = auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)
        var /= n1 * n2
        ci_lower = auc - z * np.sqrt(var)
        ci_upper = auc + z * np.sqrt(var)
        return auc, ci_lower, ci_upper


    # Yuval's version (probabilities instead of binary predicitons):
    def run_pipeline(self):
        # Load and split data
        main_df = self.data_handler.load_data()
        logging.info(f'finished loading data')
        logging.info(f'main_df shape: {main_df.shape}')

        # print('main_df shape:', main_df.shape)
        X_train, X_test, y_train, y_test = self.data_handler.split_data(main_df)
        logging.info(f'X_train shape: {X_train.shape}')
        logging.info(f'X_test shape: {X_test.shape}')
        logging.info(f'y_train shape: {y_train.shape}')
        logging.info(f'y_test shape: {y_test.shape}')

        # Fit preprocessing steps on the train set
        # get a list of the x features for the model
        x_features_list = [col for col in X_train.columns if col != 'hospital_death']
        X_train_processed, fitted_scaler, feature_info_dtype, dict_of_fill_values, encoder_info = self.preprocessing.run_preprocessing_fit(data=X_train, list_of_x_features_for_model=x_features_list)
        logging.info(f'finished preprocessing fit')
        logging.info(f'feature_info_dtype: {feature_info_dtype}')
        logging.info(f'dict_of_fill_values: {dict_of_fill_values}')

        X_test_processed = self.preprocessing.run_preprocessing_transform(
            data=X_test,
            scaler=fitted_scaler,
            feature_info_dtype=feature_info_dtype,
            dict_of_fill_values=dict_of_fill_values,
            encoder_information=encoder_info
        )
        logging.info(f'finished preprocessing transform')
        logging.info('X_train_processed shape after preprocessing:', X_train_processed.shape)
        logging.info('X_test_processed shape after preprocessing:', X_test_processed.shape)

        # Train model on processed train set
        self.model_handler.train(X_train_processed, y_train)
        logging.info('finished training')
        

        # Predict on processed test set
        binary_predictions = self.model_handler.predict(X_test_processed)
        probabilities = self.model_handler.predict_proba(X_test_processed)[:, 1]  # Probabilities for the positive class
        logging.info('finished predicting')
        
        # store all the predictions in a dataframe save it to a csv file with os
        predictions_df = pd.DataFrame({'y_test':y_test,'binary_predictions': binary_predictions, 'probabilities': probabilities})
        
        

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

        

        # Assuming the rest of your script is here, especially the part where predictions_df is created

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
