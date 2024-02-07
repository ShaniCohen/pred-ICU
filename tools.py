
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

#editor_fold
fill_na_ethnic=True # fill  'Other/Unknown' to null values in ethnich feature
output_path='output/missing_data/'
drop_apache=True
mpl.rcParams['axes.titlesize'] = 16  # You can adjust the size as needed
filter_continious_features=True
drop_irrelevant_cols=['encounter_id','patient_id'] # drop irrelavnt cols


df=pd.read_csv('data/training_v2.csv').iloc[:1000,:]
target=df["hospital_death"]

###----------editor fold preprocessing-------------###

#go to fill_na_ethnic declaration
if fill_na_ethnic:
    df['ethnicity']=df['ethnicity'].fillna('Other/Unknown')

#drop apache cols
if drop_apache:
    df.drop(columns=['apache_4a_hospital_death_prob','apache_4a_icu_death_prob'],inplace=True)

#keep continious features
if filter_continious_features:
    continious_columns= df.columns[(df.nunique() >=10) & ((df.dtypes == 'int64')|(df.dtypes == 'float64'))]
    df=df.loc[:,continious_columns]

#drop irrelavnt cols
df.drop(columns=drop_irrelevant_cols)


evaluation_dictionary={}
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def bootstrap_confidence_interval(predictions, y_true, metric_func,evaluation_dictionary, n_iterations=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for a given metric.

    Parameters:
    - predictions: Model predictions.
    - y_true: True labels.
    - metric_func: Metric function to calculate (e.g., roc_auc_score).
    - n_iterations: Number of bootstrap iterations.
    - confidence_level: Confidence level for the interval.

    Returns:
    - A tuple containing the lower and upper bounds of the confidence interval.
    """
    scores = []
    for _ in range(n_iterations):
        # Bootstrap sample indices
        indices = np.random.choice(range(len(predictions)), size=len(predictions), replace=True)
        # Calculate and store the score
        score = metric_func(y_true.iloc[indices], predictions[indices])
        scores.append(score)

    # Calculate the confidence interval
    lower = np.percentile(scores, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(scores, (1 + confidence_level) / 2 * 100)

    return lower, upper

def eval_technique_with_bootstrap(x_train, y_train, x_test, y_test, technique_name, n_iterations=1000):
    models = {
        'Random Forest': RandomForestClassifier(max_depth=4,n_estimators= 10),
        'Logistic Regression': LogisticRegression(max_iter=10)
    }
    
    # Dictionary to store results
    results = {technique_name: {}}
    x_train=x_train.reset_index(drop=True)
    x_test=x_test.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)
    y_test=y_test.reset_index(drop=True)


    for model_name, model in models.items():
        model.fit(x_train, y_train)
        
        # For metrics requiring probabilities
        if hasattr(model, "predict_proba"):
            predictions_proba=model.predict_proba(x_test.iloc[:,:])[:,1]
            auc_ci = bootstrap_confidence_interval(predictions_proba, y_test, roc_auc_score, n_iterations)
            results[technique_name][f'{model_name} AUC CI'] = auc_ci
        
        # For all metrics
        predictions = model.predict(x_test)
        accuracy_ci = bootstrap_confidence_interval(predictions, y_test, accuracy_score, n_iterations)
        f1_ci = bootstrap_confidence_interval(predictions, y_test, f1_score, n_iterations)
        
        results[technique_name][f'{model_name} Accuracy CI'] = accuracy_ci
        results[technique_name][f'{model_name} F1 Score CI'] = f1_ci
    
        # Update the global dictionary
    evaluation_dictionary.update(results)
    return results
    


    # Original DataFrame shape
original_shape = df.shape

# Step 1: Drop columns where more than 50% of the values are null
threshold = len(df) * 0.5  # 50% of the number of rows
df_drop_nulls = df.dropna(thresh=threshold, axis=1)

# Step 2: Remove any rows that still have null values
df_drop_nulls.dropna(inplace=True)

# New DataFrame shape after dropping nulls
new_shape = df_drop_nulls.shape

print(f"Original DataFrame shape: {original_shape}")
print(f"Shape after dropping columns with >50% nulls and rows with any nulls: {new_shape}")

X=df_drop_nulls.reset_index(drop=True)
y=target[df_drop_nulls.index].reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
eval_technique_with_bootstrap(X_train, y_train, X_test, y_test, technique_name="drop_null", n_iterations=10)
print(evaluation_dictionary)