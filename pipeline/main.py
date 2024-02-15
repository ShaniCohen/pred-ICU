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


# editor_fold
seed = 1
reg_alpha_param = 0.2


# Main function of the pred-ICU Pipeline
def main(model):
    # Setup logging
    setup_logging()

    # Create objects
    data_handler = DataHandler(file_path=os.path.abspath('..\\data\\training_v2.csv'))
    preprocessing = Preprocessing(scaler=MinMaxScaler())
    model_handler = ModelHandler(model=model)

    # Create pipeline object
    pipeline = MLClassificationPipeline(data_handler=data_handler,
                                        preprocessing=preprocessing,
                                        model_handler=model_handler,
                                        number_of_splits=5)

    results_file_path = pipeline.run_pipeline()

    logging.info(f'finished main execution\n')
    return results_file_path


if __name__ == '__main__':
    # run main on a list of models
    # models = []
    # models.append(LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=seed))
    # models.append(xgb.XGBClassifier(random_state=seed, reg_alpha=reg_alpha_param))
    # models.append(RandomForestClassifier(random_state=seed))

    # list_of_results_file_paths = [main(model) for model in models]
    logistic_regression_results_file_path = os.path.abspath('../predictions/logistic_regression_initial_results.json')
    xgb_results_file_path = os.path.abspath('../predictions/xgb_initial_results.json')
    random_forest_results_file_path = os.path.abspath('../predictions/random_forest_initial_results.json')
    list_of_results_file_paths = []
    list_of_results_file_paths.append(logistic_regression_results_file_path)
    list_of_results_file_paths.append(xgb_results_file_path)
    list_of_results_file_paths.append(random_forest_results_file_path)

    # Create ModelEvaluation object
    model_evaluation = ModelEvaluation(json_files=list_of_results_file_paths)

    # Plot ROC curves
    # model_evaluation.plot_roc_curves(including_apache=True, including_cutoffs=False, including_confidence_intervals=False)

    # Plot Precision-Recall curves
    # model_evaluation.plot_precision_recall_curves(including_apache=True, including_cutoffs=False, including_confidence_intervals=False)

    # Plot Sensitivity-Percent-Positives curves
    model_evaluation.plot_sensitivity_percent_positives_curves(including_apache=True, including_cutoffs=False, including_confidence_intervals=False)

    # Plot Precision-Percent-Positives curves
    # model_evaluation.plot_precision_percent_positives_curves(including_apache=True, including_cutoffs=False, including_confidence_intervals=False)

    # Generate predictions .csv files for calibrations plots
    # model_evaluation.generate_predictions_files()
