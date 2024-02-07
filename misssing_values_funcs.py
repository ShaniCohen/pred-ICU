import pandas as pd
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb 
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu, pointbiserialr
from scipy.stats import chi2_contingency
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import math
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer


def spec_smart_fill_na(x_train, x_test, y_train, y_test):
    for x in [x_train, x_test]:
        x['ethnicity']=x['ethnicity'].fillna('Other/Unknown')
        x['bmi'] = x.apply(lambda row: row['weight'] / ((row['height']/100) ** 2) if pd.isnull(row['bmi']) else row['bmi'], axis=1)
        

def complete_case_analysis(x_train, x_test, y_train, y_test, thresh=0.5):
    # Identify columns in X_train with more than 'thresh' proportion of null values
    cols_to_drop = x_train.columns[x_train.isnull().mean() > thresh]
    
    # Drop identified columns from X_train and X_test
    x_train = x_train.drop(columns=cols_to_drop)
    x_test = x_test.drop(columns=cols_to_drop)
    
    # Drop rows with any null values in X_train and X_test
    x_train = x_train.dropna()
    x_test = x_test.dropna()
    
    # Since rows are dropped from X_train and X_test, we need to ensure y_train and y_test are aligned
    # This means dropping the corresponding rows in y_train and y_test
    y_train = y_train.loc[x_train.index]
    y_test = y_test.loc[x_test.index]
    
    #validation
    if ((x_test.isna().sum().sum()>0)|( x_train.isna().sum().sum()>0)):
        print("Error: there are null vlaues")
    return x_train, x_test, y_train, y_test



def impute_central_tendency(x_train, x_test, y_train, y_test, impute_strategy='median'):

    """
    Fills missing values in x_train and x_test.
    Continuous features are imputed with either the median or mean as specified.
    Categorical features are imputed with the mode.
    
    Parameters:
    - x_train: Training features DataFrame.
    - x_test: Testing features DataFrame.
    - y_train: Training target Series or DataFrame.
    - y_test: Testing target Series or DataFrame.
    - impute_strategy: String, either 'median' or 'mean', specifying how to impute continuous features.
    
    Returns:
    - x_train, x_test, y_train, y_test: DataFrames or Series with missing values imputed.
    """

    # Impute missing values in X_train
    for column in x_train.columns:
        if x_train[column].dtype == 'float64' or x_train[column].dtype == 'int64':
            # Determine the fill value based on the impute_strategy
            if impute_strategy == 'median':
                fill_value = x_train[column].median()
            elif impute_strategy == 'mean':
                fill_value = x_train[column].mean()
            else:
                raise ValueError("Invalid impute_strategy. Choose 'mean' or 'median'.")
            
            x_train[column].fillna(fill_value, inplace=True)
            x_test[column].fillna(fill_value, inplace=True)  # Apply the same fill value to X_test
        else:
            # For categorical features, use mode
            fill_value = x_train[column].mode()[0]
            x_train[column].fillna(fill_value, inplace=True)
            x_test[column].fillna(fill_value, inplace=True)  # Apply the same fill value to X_test
    
    # Check for any remaining null values in X_train or X_test
    if (x_test.isna().sum().sum() > 0) or (x_train.isna().sum().sum() > 0):
        print("Error: there are null values")
        
    return x_train, x_test, y_train, y_test


def stochastic_imputation(x_train, x_test, y_train, y_test):
    """
    Stochastically imputes missing values in x_train and x_test.
    Categorical features: Missing values are filled by resampling observed category levels.
    Continuous features: resample observed values directly

    Parameters:
    - x_train, x_test: Training and testing features DataFrame.
    - y_train, y_test: Training and testing target Series or DataFrame.

    Returns:
    - Modified x_train, x_test, y_train, y_test with imputed values.
    """

    for column in x_train.columns:
        if x_train[column].dtype == 'float64' or x_train[column].dtype == 'int64':  # Continuous
            observed_values = x_train[column].dropna()
            x_train[column] = x_train[column].apply(lambda x: np.random.choice(observed_values) if pd.isnull(x) else x)
            x_test[column] = x_test[column].apply(lambda x: np.random.choice(observed_values) if pd.isnull(x) else x)
        else:  # Categorical
            # Resample observed category levels for both train and test
            observed_categories = x_train[column].dropna()
            x_train[column] = x_train[column].apply(lambda x: np.random.choice(observed_categories) if pd.isnull(x) else x)
            x_test[column] = x_test[column].apply(lambda x: np.random.choice(observed_categories) if pd.isnull(x) else x)


        # Check for any remaining null values in X_train or X_test
    if (x_test.isna().sum().sum() > 0) or (x_train.isna().sum().sum() > 0):
        print("Error: there are null values")
    # Assumption: y_train and y_test do not require imputation.
    return x_train, x_test, y_train, y_test

def flag_imputation(x_train, x_test, y_train, y_test,flag=-1):
    


    # single column with negastive number
    x_train['pre_icu_los_days'].fillna(-100,inplace=True)
    x_test['pre_icu_los_days'].fillna(-100,inplace=True)
    x_train.fillna(flag,inplace=True)
    x_test.fillna(flag,inplace=True)
        # Check for any remaining null values in X_train or X_test
    if (x_test.isna().sum().sum() > 0) or (x_train.isna().sum().sum() > 0):
        print("Error: there are null values")
    return x_train, x_test, y_train, y_test


def multiple_imputation(x_train, x_test, y_train, y_test,max_iter=10):
    """
    This function performs imputation on a dataset that contains both numerical and categorical features,
      splitting the process into two distinct strategies for each data type.
        It's designed to work with training and testing sets,
          preparing them for machine learning models by filling in missing values.

    Parameters:
    x_train (DataFrame): The training dataset with features, possibly containing missing values.
    x_test (DataFrame): The testing dataset with features, possibly containing missing values.
    y_train (Series or DataFrame): The target variable for the training dataset.
    y_test (Series or DataFrame): The target variable for the testing dataset.
    max_iter (int, optional): The maximum number of iterations for the IterativeImputer to impute missing values in numerical columns. Default is 1, indicating single imputation.
    Returns:
    x_train_imputed (DataFrame): The training dataset with missing values imputed.
    x_test_imputed (DataFrame): The testing dataset with missing values imputed.
    y_train, y_test: The target variables, returned without modification.
    """

    # Separately handle categorical and numerical columns
    categorical_cols = x_train.columns[x_train.dtypes == 'object']
    numerical_cols = x_train.columns[x_train.dtypes != 'object']
    
    # Apply IterativeImputer on numerical columns
    num_imputer = IterativeImputer(random_state=100, max_iter=max_iter)
    x_train_num = pd.DataFrame(num_imputer.fit_transform(x_train[numerical_cols]), columns=numerical_cols)
    x_test_num = pd.DataFrame(num_imputer.transform(x_test[numerical_cols]), columns=numerical_cols)
    
    # Apply SimpleImputer on categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    x_train_cat = pd.DataFrame(cat_imputer.fit_transform(x_train[categorical_cols]), columns=categorical_cols)
    x_test_cat = pd.DataFrame(cat_imputer.transform(x_test[categorical_cols]), columns=categorical_cols)
    
    # Concatenate the imputed columns back together
    x_train_imputed = pd.concat([x_train_num, x_train_cat], axis=1)
    x_test_imputed = pd.concat([x_test_num, x_test_cat], axis=1)

    if (x_train_imputed.isna().sum().sum() > 0) or (x_test_imputed.isna().sum().sum() > 0):
        print("Error: there are null values")

    return x_train_imputed, x_test_imputed, y_train, y_test

#editor_fold
fill_na_ethnic=True # fill  'Other/Unknown' to null values in ethnich feature
output_path='output/missing_data/'
drop_apache=True
mpl.rcParams['axes.titlesize'] = 16  # You can adjust the size as needed
filter_continious_features=False
drop_irrelevant_cols=['encounter_id','patient_id','hospital_death',
                      'apache_4a_hospital_death_prob','apache_4a_icu_death_prob','readmission_status'] # drop irrelavnt cols



df=pd.read_csv('data/training_v2.csv')
target=df["hospital_death"]

###----------editor fold preprocessing-------------###

#go to fill_na_ethnic declaration
if fill_na_ethnic:
    df['ethnicity']=df['ethnicity'].fillna('Other/Unknown')


#keep continious features
if filter_continious_features:
    continious_columns= df.columns[(df.nunique() >=10) & ((df.dtypes == 'int64')|(df.dtypes == 'float64'))]
    df=df.loc[:,continious_columns]

#drop irrelavnt cols
df.drop(columns=drop_irrelevant_cols,inplace=True)

X_train, X_test, y_train, y_test = train_test_split( df, target, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test=multiple_imputation(X_train, X_test, y_train, y_test)

