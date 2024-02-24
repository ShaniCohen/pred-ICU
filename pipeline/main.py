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
import time

# Editor_fold
seed = 1
reg_alpha_param = 0.2
np.random.seed(42)

for i in range(100):
    # Main function of the pred-ICU Pipeline
    def main(model):
        # Setup logging
        setup_logging()
        start_time = time.time()  # Start time

        # Create objects
        data_handler = DataHandler(file_path=os.path.abspath('..\\data\\training_v2.csv'))
        preprocessing = Preprocessing(scaler=MinMaxScaler())
        model_handler = ModelHandler(model=model)
        impute = Impute('impute_central_tendency')  # no_imputation

        # Create pipeline object
        pipeline = MLClassificationPipeline(data_handler=data_handler,
                                            preprocessing=preprocessing,
                                            model_handler=model_handler,
                                            impute=impute,
                                            number_of_splits=5,
                                            do_shap=False,
                                            to_scale=False,
                                            to_optimize_hyperparams=False,
                                            seed=seed)

        results_file_path = pipeline.run_pipeline()

        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        logging.info(f'Finished main execution in {elapsed_time} seconds\n')
        print(f'Finished main execution in {elapsed_time} seconds\n')
        return results_file_path


    if __name__ == '__main__':
        
        # run main on a list of models
        models = []

    # models.append(LogisticRegression(penalty='l1', solver='saga', max_iter=100, random_state=seed))
    # LogisticRegression Best Params: {'C': 40.520133538437136, 'penalty': 'l2', 'solver': 'liblinear'}

    models.append(xgb.XGBClassifier(random_state=seed,
                                    alpha=reg_alpha_param,
                                    eval_metric='logloss',
                                    early_stopping_rounds=10,
                                    verbose=0,
                                    verbose_eval=False))
    # XGBClassifier Best Params: {'learning_rate': 0.3254250016856113, 'n_estimators': 107, 'max_depth': 4, 'subsample': 0.9362235826133668, 'colsample_bytree': 0.8283618954532366}

    # models.append(RandomForestClassifier(random_state=seed))
    # RandomForestClassifier Best Params: {'n_estimators': 969, 'max_depth': 11, 'min_samples_split': 18, 'min_samples_leaf': 8, 'max_features': None}

    # list_of_results_file_paths = [main(model) for model in models]
    # ------------------------------------------------------------
    
    # list_of_results_file_paths = ['C:\\Users\\nirro\\Desktop\\MSc\\predictive_modeling_healthcare\\git\pred-ICU\\predictions\\predictions_2024-02-19_15-05-57.json']
    list_of_results_file_paths = [os.path.abspath('..\\predictions\\xgb_first_seed_results.json')]

    # Create ModelEvaluation object
    model_evaluation = ModelEvaluation(json_files=list_of_results_file_paths)

    # Plot ROC Curves
    # model_evaluation.plot_roc_curves(including_apache=True, including_cutoffs=True, including_confidence_intervals=True)

    # Plot Precision-Recall Curves
    # model_evaluation.plot_precision_recall_curves(including_apache=True, including_cutoffs=False, including_confidence_intervals=False)

    # Plot Sensitivity-Percent-Positives Curves
    # model_evaluation.plot_sensitivity_percent_positives_curves(including_apache=False, including_cutoffs=True, including_confidence_intervals=False)

    # Plot Precision-Percent-Positives Curves
    # model_evaluation.plot_precision_percent_positives_curves(including_apache=False, including_cutoffs=True, including_confidence_intervals=False)

    # Plot Decile Patient Prediction Uncertainty Graph
    # model_evaluation.plot_decile_patient_prediction_uncertainty(lowest=True, middle=True, highest=True)

    # Generate Predictions .csv Files for Calibrations Plots
    # model_evaluation.generate_predictions_files()

        # # plot shap
        # base_directory = os.path.dirname(os.path.dirname(DataHandler(file_path=os.path.abspath('..\\data\\training_v2.csv')).file_path))
        # # print(base_directory)
        # shap_directory = os.path.join(base_directory, 'shap_values')
        # corrent_file_path = os.path.join(shap_directory, 'shap_values_2024-02-19_15-05-55.pkl')
        # # print(corrent_file_path)
        # # print(os.path.exists(corrent_file_path))
        # model_evaluation.plot_shap_waterfall(shap_values_path=corrent_file_path, instance_index=0)
