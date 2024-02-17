import numpy as np
from logging_config import setup_logging
import os
import logging
from data_handler import DataHandler
from preprocessing import Preprocessing
from sklearn.preprocessing import MinMaxScaler
from model_handler import ModelHandler
from ml_classification_pipeline import MLClassificationPipeline
from model_evaluation import ModelEvaluation
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from misssing_values_funcs import Impute


# Editor_fold
seed = 1
reg_alpha_param = 0.2
np.random.seed(42)


# Main function of the pred-ICU Pipeline
def main(model):
    # Setup logging
    setup_logging()

    # Create objects
    data_handler = DataHandler(file_path=os.path.abspath('..\\data\\training_v2.csv'))
    preprocessing = Preprocessing(scaler=MinMaxScaler())
    model_handler = ModelHandler(model=model)
    impute = Impute('no_imputation')

    # Create pipeline object
    pipeline = MLClassificationPipeline(data_handler=data_handler,
                                        preprocessing=preprocessing,
                                        model_handler=model_handler,
                                        impute=impute,
                                        number_of_splits=5)

    results_file_path = pipeline.run_pipeline()

    logging.info(f'finished main execution\n')
    return results_file_path


if __name__ == '__main__':
    
    # run main on a list of models
    models = []

    models.append(LogisticRegression(penalty='l1', solver='saga', max_iter=100, random_state=seed))
    # model name: LogisticRegression best params: {'C': 40.520133538437136, 'penalty': 'l2', 'solver': 'liblinear'}

    models.append(xgb.XGBClassifier(random_state=seed, alpha=reg_alpha_param,eval_metric='logloss', early_stopping_rounds=10))
    # model name: XGBClassifier best params: {'learning_rate': 0.3254250016856113, 'n_estimators': 107, 'max_depth': 4, 'subsample': 0.9362235826133668, 'colsample_bytree': 0.8283618954532366}

    models.append(RandomForestClassifier(random_state=seed))
    # model name: RandomForestClassifier best params: {'n_estimators': 969, 'max_depth': 11, 'min_samples_split': 18, 'min_samples_leaf': 8, 'max_features': None}

    list_of_results_file_paths = [main(model) for model in models]
    # ------------------------------------------------------------
    
    # list_of_results_file_paths = ['C:\\Users\\nirro\\Desktop\\MSc\\predictive_modeling_healthcare\\git\pred-ICU\\predictions\\predictions_2024-02-13_22-21-39.json']
    # Create ModelEvaluation object
    model_evaluation = ModelEvaluation(json_files=list_of_results_file_paths)

    # Plot ROC curves
    model_evaluation.plot_roc_curves(including_apache=True, including_cutoffs=True, including_confidence_intervals=True)

    # Plot Precision-Recall curves
    # model_evaluation.plot_precision_recall_curves(including_apache=True, including_cutoffs=False, including_confidence_intervals=False)

    # Plot Sensitivity-Percent-Positives curves
    # model_evaluation.plot_sensitivity_percent_positives_curves(including_apache=False, including_cutoffs=True, including_confidence_intervals=False)

    # Plot Precision-Percent-Positives curves
    # model_evaluation.plot_precision_percent_positives_curves(including_apache=False, including_cutoffs=True, including_confidence_intervals=False)

    # Generate predictions .csv files for calibrations plots
    # model_evaluation.generate_predictions_files()
