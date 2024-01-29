import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import xgboost as xgb
import joblib


class ModelHandler:
  def __init__(self, model):
      self.model = model
  def save_predictions(self, x_features, model_configurations,predictions, file_name):
      
    predictions.to_csv(file_name, index=False)
        
  def train(self, X_train, y_train):
      model_configurations = self.model.get_params()
      model_seed = 42
      
      self.model.fit(X_train, y_train)
      
  
  def predict(self, X_test):
      return self.model.predict(X_test)
