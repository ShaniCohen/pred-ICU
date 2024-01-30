from logging_config import setup_logging
from data_handler import DataHandler
import os
from preprocessing import Preprocessing
from sklearn.preprocessing import MinMaxScaler
from model_handler import ModelHandler
import xgboost as xgb
from ml_classification_pipeline import MLClassificationPipeline
import logging
from model_evaluation import ModelEvaluation
import json
from sklearn.linear_model import LogisticRegression
# import tabnet
from pytorch_tabnet.tab_model import TabNetClassifier

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



# Main function of the pred-ICU Pipeline
def main(model):
    # Setup logging
    setup_logging()

    # Create objects
    data_handler = DataHandler(file_path=os.path.abspath('..\\data\\training_v2.csv'))
    preprocessing = Preprocessing(scaler=MinMaxScaler())
    model_handler = ModelHandler(model=model)

    # Create pipeline object
    pipeline = MLClassificationPipeline(data_handler=data_handler, preprocessing=preprocessing, model_handler=model_handler)

    # Run pipeline
    results_file_path = pipeline.run_pipeline()
    
    
    logging.info(f'finished main execution\n')
    return results_file_path

if __name__ == '__main__':
    # run main on a list of models
    lr_model = LogisticRegression()
    xgb_model = xgb.XGBClassifier()
    tabnet_model = TabNetClassifier()
    models = [lr_model,xgb_model,tabnet_model]
    list_of_results_file_paths = []
    for model in models:
        list_of_results_file_paths.append(main(model))

    # list_of_results_file_paths = [os.path.abspath('..\\predictions\\predictions_2024-01-30_20-21-10.json'),
    #                               os.path.abspath('..\\predictions\\predictions_2024-01-30_20-21-19.json')]
    # # Create ModelEvaluation object
    model_evaluation = ModelEvaluation(json_files=list_of_results_file_paths)
    # Plot ROC curves
    model_evaluation.plot_auc_curves()
    
    # plot precision-recall curves
    model_evaluation.plot_precision_recall_curves()
    
    # Calculate metrics
    list_of_thresholds = [0.7,0.8,0.85,0.9,0.95]
    metrics = model_evaluation.calculate_metrics_with_bootstrap(list_of_thresholds, n_bootstrap=100)
    print(metrics)
    model_evaluation.plot_metrics_with_ci(metrics)

    



            
