        
        # Import libraries
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, auc, classification_report,
                                confusion_matrix, f1_score, precision_score,
                                recall_score, roc_curve)

class ModelEvaluation:
    def __init__(self,file_path):
        self.file_path = file_path
        # open the results file and read it, the file is a dictionary
        # json_object = {f'{model_name}_{str(model_params)}': predictions_dict}
        self.results = pd.read_json(self.file_path, orient='index')
        # create a dataframe from the dictionary
        self.results_df = pd.DataFrame.from_dict(self.results.iloc[0,0])
        self.y_test = self.results_df['y_test'].astype('int')
        self.binary_predictions = self.results_df['binary_predictions'].astype('int')
        self.probabilities = self.results_df['probabilities']
        self.model_name = self.results.index[0].split('_')[0]
        self.model_params = self.results.index[0].split('_')[1]
        self.model_params = self.model_params.replace('(', '').replace(')', '').replace("'", '').replace(', ', '_')
        self.model_params = self.model_params.replace(' ', '').replace(':', '_').replace('.', '_')


    
    
    # Evaluation Metrics
    logging.info(f'classification_report: \n{classification_report(self.y_test, self.binary_predictions)}')
    logging.info(f'precision_score: {precision_score(self.y_test, self.binary_predictions)}')
    logging.info(f'recall_score: {recall_score(self.y_test, self.binary_predictions)}')
    logging.info(f'f1_score: {f1_score(self.y_test, self.binary_predictions)}')


    def plot_ROC_Curve_calibration_curve():
        # Calculate AUROC
        roc_auc, ci_lower, ci_upper = self.compute_auc_confidence_interval(self.y_test, self.probabilities)
        logging.info(f'calculate AUROC of the model: {roc_auc}')
        fpr, tpr, thresholds = roc_curve(self.y_test, self.probabilities)

        plt.figure(figsize=(10, 10))

        # Subplot 1 for ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(fpr, tpr, label='AUROC = %0.3f (%0.3f - %0.3f)' % (roc_auc, ci_lower, ci_upper))
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC Curve')
        plt.legend(handlelength=0)
        # Subplot 2 for Calibration Curve# Compute the calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, self.probabilities, n_bins=10)

        # Subplot 2 for Calibration Plot
        plt.subplot(1, 2, 2)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-")
        plt.plot([0, 1], [0, 1], "k:")
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.title('Calibration Plot')
        plt.legend(handlelength=0)

        # Adjust layout
        plt.tight_layout()
        plt.show()