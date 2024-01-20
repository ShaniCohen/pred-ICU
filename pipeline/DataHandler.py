import pandas as pd
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        # Load icu data from file_path
        path_to_icu_data = self.file_path
        main_df = pd.read_csv(path_to_icu_data)
        return main_df
    
    def eda(self, data):
        # Perform exploratory data analysis
        # Implement your EDA logic here
        print(f'EDA on data: \n{data.head()}')
        print(f'desctibe: \n{data.describe()}')
        print(f'info: \n{data.info()}')
        print(f'columns: \n{data.columns}')
        print(f'nulls: \n{data.isnull().sum()}')
        # maybe we can add some plots here
        pass

    def split_data(self, data, test_size=0.2):
        # Split data into training and testing sets, 
        # because we have imbalanced data we want the train and the test will have the same ratio of the target
        X = data.drop('hospital_death',axis=1)
        y = data['hospital_death']
        # drop ID columns 
        X = X.drop(['encounter_id','patient_id','hospital_id'],axis=1)
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Implement your data splitting logic here
        return X_train, X_test, y_train, y_test

