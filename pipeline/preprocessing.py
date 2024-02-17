from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import logging


class Preprocessing:
    def __init__(self, scaler):
        self.scaler = scaler  # Example for normalization
        # Other attributes for preprocessing

    def get_feature_info(self, data, list_of_x_features_for_model):
        # Identify and store column types
        # Implement logic to find types and column names
        dict_of_features_dtypes = {feature: data[feature].dtype for feature in list_of_x_features_for_model}
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
    
    def run_preprocessing_fit(self, data, list_of_x_features_for_model,to_scale):
        # Fit preprocessing steps like normalization, missing value imputation, etc., on the train set
        # 1. Identify and store column names and data types
        # create a dictionary with the column names and the data types
        feature_info_dtype = self.get_feature_info(data, list_of_x_features_for_model)
        logging.info(f'feature_info_dtype: {feature_info_dtype}')
        # 2. Fill missing values
        dict_of_fill_values = self.get_fill_values_dict(data)
        data = data.fillna(dict_of_fill_values)
        
        logging.info(f'fill values: {dict_of_fill_values}')
        # 3. Feature extraction and engineering
        # 3.5 categorical encoding
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
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
        
        if to_scale:
            # 4. Normalize/Scale data
            scaled_data = self.scaler.fit_transform(encoded_data)  # Fitting a scaler
            # print(f'scaled_data shape: {scaled_data.shape}')
        else:
            scaled_data = encoded_data
        # 5. Feature selection
        
        return scaled_data, self.scaler, feature_info_dtype, dict_of_fill_values, encoder_info

    def run_preprocessing_transform(self, data, scaler, feature_info_dtype, dict_of_fill_values, encoder_information, to_scale):
        # transform the preprocessing steps like normalization, missing value imputation, etc., on the train and test set
        # 1. make sure the data is at same format as the train data                
        
        # print pandas version
        # Ensure test data has the same columns
        data = data.reindex(feature_info_dtype.keys(),axis=1)
        # Reorder and add missing columns with NaNs
        # expected_columns = list(feature_info_dtype.keys())
        # data = data.reindex(expected_columns)

        
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
