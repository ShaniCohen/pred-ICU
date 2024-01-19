from os import path
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
# import xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
# import precision, recall, f1_score, support
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

class Preprocessing:
    def __init__(self, scaler):
        self.scaler = scaler  # Example for normalization
        # Other attributes for preprocessing
    def get_feature_info(self, data,list_of_x_features_for_model):
        # Identify and store column types
        # Implement logic to find types and column names
        dict_of_features_dtypes = {feature:data[feature].dtype for feature in list_of_x_features_for_model}
        return dict_of_features_dtypes
    
    def get_fill_values_dict(self, data):
        # create a dictionary with the values to fill the missing values
        # for columns with numeric values we will fill with the median
        # for columns with categorical values we will fill with the mode
        dict_of_fill_values = {}
        for column in data.columns:
            if data[column].dtype == 'object':
                dict_of_fill_values[column] = data[column].mode()[0]
            else:
                dict_of_fill_values[column] = data[column].median()
        return dict_of_fill_values
    
    def run_preprocessing_fit(self, data, list_of_x_features_for_model):
        # Fit preprocessing steps like normalization, missing value imputation, etc., on the train set
        # 1. Identify and store column anmes and data types
        # create a dictionary with the column names and the data types
        feature_info_dtype = self.get_feature_info(data,list_of_x_features_for_model)
        print(feature_info_dtype)
        # 2. Fill missing values
        dict_of_fill_values = self.get_fill_values_dict(data)
        data = data.fillna(dict_of_fill_values)
        
        print(dict_of_fill_values)
        # 3. Feature extraction and engineering
        # 3.5 categorical encoding
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

        encoder = OneHotEncoder(sparse=False)
        encoded_category_data = encoder.fit_transform(data[categorical_columns])
        # Save Encoder with Column Names and Prefixes
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoder_info = {
            'encoder': encoder,
            'columns': encoded_columns
        }

        # Concatenate with the rest of the test data
        encoded_data = pd.concat([data.drop(categorical_columns, axis=1).reset_index(drop=True), pd.DataFrame(encoded_category_data, columns=encoded_columns).reset_index(drop=True)], axis=1)
        # print(f'encoded_data shape: {encoded_data.shape}')
        # 4. Normalize/Scale data
        scaled_data = self.scaler.fit_transform(encoded_data)  # Fitting a scaler
        # print(f'scaled_data shape: {scaled_data.shape}')
        
        # 5. Feature selection
        
        return scaled_data, self.scaler, feature_info_dtype, dict_of_fill_values,encoder_info

    def run_preprocessing_transform(self, data,scaler, feature_info_dtype, dict_of_fill_values,encoder_information):
        # transform the preprocessing steps like normalization, missing value imputation, etc., on the trainand test set
        # 1. make sure the data is at same format as the train data
        # Ensure test data has the same columns
        data = data.reindex(columns=feature_info_dtype.keys())
        # Cast data types
        for col, dtype in feature_info_dtype.items():
            data[col] = data[col].astype(dtype)
                
        # 2. Fill missing values
        data.fillna(dict_of_fill_values, inplace=True)
        
        # 3. Feature extraction and engineering
        
        # 3.5 categorical encoding
        encoder = encoder_information['encoder']
        encoded_columns = encoder_information['columns']

        # Apply the encoder to the test data
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        encoded_test_data = pd.DataFrame(encoder.transform(data[categorical_columns]), columns=encoded_columns).reset_index(drop=True)

        # Concatenate with the rest of the test data
        data = pd.concat([data.drop(categorical_columns, axis=1).reset_index(drop=True), encoded_test_data], axis=1)
        # 4. Normalize/Scale data
        data_scaled = scaler.transform(data)  # Scaling the data
        
        # 5. Feature selection
        return data_scaled


class ModelHandler:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


class MLClassificationPipeline:
    def __init__(self,data_handler: DataHandler, preprocessing: Preprocessing, model_handler: ModelHandler):
        self.data_handler = data_handler
        self.preprocessing = preprocessing
        self.model_handler = model_handler
        

    def run_pipeline(self):
        # Load and split data
        main_df = self.data_handler.load_data()
        # print('main_df shape:', main_df.shape)
        X_train, X_test, y_train, y_test = self.data_handler.split_data(main_df)
        # print('X_train shape before preprocessing:', X_train.shape)
        # print('X_test shape before preprocessing:', X_test.shape)
        # print('y_train shape before preprocessing:', y_train.shape)
        # print('y_test shape before preprocessing:', y_test.shape)
        
        # Fit preprocessing steps on the train set
        # get a list of the x features for the model
        x_features_list = [col for col in X_train.columns if col != 'hospital_death']
        X_train_processed,fited_scaler, feature_info_dtype, dict_of_fill_values,encoder_info = self.preprocessing.run_preprocessing_fit(data=X_train, 
                                                                                                                      list_of_x_features_for_model=x_features_list)

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

        # Check shapes of X_train and y_train
        # print("X_train shape:", X_train_processed.shape)
        # print("y_train shape:", y_train.shape)
        # print("X_test shape:", X_test_processed.shape)
        # print("y_test shape:", y_test.shape)
        
        # Train model on processed train set
        self.model_handler.train(X_train_processed, y_train)

        # Predict and evaluate on processed test set
        predictions = self.model_handler.predict(X_test_processed)
        # Add evaluation metrics here
        # Calculate precision, recall, f1-score, and support
        print(f'classification_report \n{classification_report(y_test, predictions)}')
        print(f'precision_score \n{precision_score(y_test, predictions)}')
        print(f'recall_score \n{recall_score(y_test, predictions)}')
        print(f'f1_score \n{f1_score(y_test, predictions)}')
        # plot the roc auc curve
        # calculate roc auc
        roc_auc = roc_auc_score(y_test, predictions)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        # plot no skill
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        # show the plot
        plt.show()
        

        
        
        return predictions  # or any evaluation metric results


if __name__ == '__main__':
    # Create objects
    data_handler = DataHandler(file_path=r'C:\Users\nirro\Desktop\MSc\predictive_modeling_healthcare\data\training_v2.csv')
    preprocessing = Preprocessing(scaler=MinMaxScaler())
    model_handler = ModelHandler(model=xgb.XGBClassifier())

    # Create pipeline object
    pipeline = MLClassificationPipeline(data_handler=data_handler, preprocessing=preprocessing, model_handler=model_handler)

    # Run pipeline
    predictions = pipeline.run_pipeline()
    print(predictions)
