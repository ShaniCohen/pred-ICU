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

# Main function of the pred-ICU Pipeline
def main(model=None):
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
    models = [LogisticRegression,xgb]
    list_of_results_file_paths = []
    for model in models:
        list_of_results_file_paths.append(main(model))

    # Create ModelEvaluation object
    model_evaluation = ModelEvaluation(json_files=list_of_results_file_paths)
    # Plot ROC curves
    model_evaluation.plot_auc_curves()
    # Calculate metrics
    metrics = model_evaluation.calculate_metrics(threshold=0.5)
    # Save metrics to csv
    # metrics.to_csv('metrics.csv', index=False)
    # Print metrics
    print(metrics)
    
    main()
