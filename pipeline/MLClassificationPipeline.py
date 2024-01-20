from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve,auc
import matplotlib.pyplot as plt
import pandas as pd 
from DataHandler import DataHandler
from Preprocessing import Preprocessing
from ModelHandler import ModelHandler
import logging
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class MLClassificationPipeline:
    def __init__(self,data_handler: DataHandler, preprocessing: Preprocessing, model_handler: ModelHandler):
        self.data_handler = data_handler
        self.preprocessing = preprocessing
        self.model_handler = model_handler
        

    def run_pipeline(self):
        # Load and split data
        main_df = self.data_handler.load_data()
        logger.info(f'finished loading data')
        logger.info(f'main_df shape: {main_df.shape}')
        
        # print('main_df shape:', main_df.shape)
        X_train, X_test, y_train, y_test = self.data_handler.split_data(main_df)
        logger.info(f'X_train shape: {X_train.shape}')
        logger.info(f'X_test shape: {X_test.shape}')
        logger.info(f'y_train shape: {y_train.shape}')
        logger.info(f'y_test shape: {y_test.shape}')
        
        # Fit preprocessing steps on the train set
        # get a list of the x features for the model
        x_features_list = [col for col in X_train.columns if col != 'hospital_death']
        X_train_processed,fited_scaler, feature_info_dtype, dict_of_fill_values,encoder_info = self.preprocessing.run_preprocessing_fit(data=X_train, 
                                                                                                        list_of_x_features_for_model=x_features_list)
        logger.info(f'finished preprocessing fit')
        logger.info(f'feature_info_dtype: {feature_info_dtype}')
        logger.info(f'dict_of_fill_values: {dict_of_fill_values}')
        
        # # Transform both train and test sets
        # X_train_processed = self.preprocessing.run_preprocessing_transform(data=X_train,
        #                                                                    scaler=fited_scaler, 
        #                                                                    feature_info_dtype=feature_info_dtype, 
        #                                                                    dict_of_fill_values=dict_of_fill_values,
        #                                                                    encoder_information=encoder_info)
        X_test_processed = self.preprocessing.run_preprocessing_transform(data=X_test,
                                                                          scaler=fited_scaler, 
                                                                          feature_info_dtype=feature_info_dtype, 
                                                                          dict_of_fill_values=dict_of_fill_values,
                                                                          encoder_information=encoder_info)
        logger.info(f'finished preprocessing transform')
        logger.info('X_train_processed shape after preprocessing:', X_train_processed.shape)
        logger.info('X_test_processed shape after preprocessing:', X_test_processed.shape)
    
        
        # Train model on processed train set
        self.model_handler.train(X_train_processed, y_train)
        logger.info(f'finished training')
        
        # Predict and evaluate on processed test set
        predictions = self.model_handler.predict(X_test_processed)
        logger.info(f'finished predicting')
        
        # Add evaluation metrics here
        # Calculate precision, recall, f1-score, and support
        # print(f'classification_report \n{classification_report(y_test, predictions)}')
        logger.info(f'classification_report \n{classification_report(y_test, predictions)}')
        # print(f'precision_score \n{precision_score(y_test, predictions)}')
        logger.info(f'precision_score \n{precision_score(y_test, predictions)}')
        # print(f'recall_score \n{recall_score(y_test, predictions)}')
        logger.info(f'recall_score \n{recall_score(y_test, predictions)}')
        # print(f'f1_score \n{f1_score(y_test, predictions)}')
        logger.info(f'f1_score \n{f1_score(y_test, predictions)}')
        # plot the roc auc curve
        # calculate roc auc
        roc_auc = roc_auc_score(y_test, predictions)
        logger.info(f'calculate AUC of model \n{roc_auc}')
        
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        # plot no skill
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        # show the plot
        plt.show()
    
        return predictions  # or any evaluation metric results

