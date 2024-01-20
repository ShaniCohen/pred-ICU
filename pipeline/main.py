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
from pipeline import DataHandler, Preprocessing, ModelHandler, MLClassificationPipeline
def main()
    # Create objects
    data_handler = DataHandler(file_path=r'C:\Users\nirro\Desktop\MSc\predictive_modeling_healthcare\data\training_v2.csv')
    preprocessing = Preprocessing(scaler=MinMaxScaler())
    model_handler = ModelHandler(model=xgb.XGBClassifier())

    # Create pipeline object
    pipeline = MLClassificationPipeline(data_handler=data_handler, preprocessing=preprocessing, model_handler=model_handler)

    # Run pipeline
    predictions = pipeline.run_pipeline()
    print(predictions)

if __name__ == '__main__':
    main()
