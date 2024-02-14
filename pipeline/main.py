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
#editor_fold
seed=1
reg_alpha_param=0.2
# Main function of the pred-ICU Pipeline
def main(model):
    # Setup logging
    setup_logging()

    # Create objects
    data_handler = DataHandler(file_path=('data/training_v2.csv'))
    preprocessing = Preprocessing(scaler=MinMaxScaler())
    model_handler = ModelHandler(model=model)
    impute=Impute('no_imputation')

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
    models.append(LogisticRegression(penalty='l1', solver='saga', max_iter=100,random_state=seed))
    models.append(xgb.XGBClassifier(random_state=seed,alpha=reg_alpha_param))
    models.append(RandomForestClassifier(random_state=seed))

    list_of_results_file_paths = [main(model) for model in models]

    # Create ModelEvaluation object
    cutoffs = [0.01, 0.05, 0.1, 0.2]
    model_evaluation = ModelEvaluation(json_files=list_of_results_file_paths, cutoffs=cutoffs)

    # Plot ROC curves
    # model_evaluation.plot_roc_curves()

    # Plot Precision-Recall curves
    # model_evaluation.plot_precision_recall_curves(including_confidence_intervals=False)

    # Plot Sensitivity-Percent-Positives curves
    # model_evaluation.plot_sensitivity_percent_positives_curves(including_confidence_intervals=False)

    # Plot Precision-Percent-Positives curves
    # model_evaluation.plot_precision_percent_positives_curves(including_confidence_intervals=False)

    # Generate predictions .csv files for calibrations plots
    model_evaluation.generate_predictions_files()

    # Calculate metrics
    # list_of_thresholds = [0.7,0.8,0.85,0.9,0.95]
    # metrics = model_evaluation.calculate_metrics_with_bootstrap(list_of_thresholds, n_bootstrap=100)
    # print(metrics)
    # model_evaluation.plot_metrics_with_ci(metrics)