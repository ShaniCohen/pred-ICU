        
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


import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score

class ModelEvaluation:
    def __init__(self, json_files):
        # Load all dataframes from json files
        self.model_results = {}
        for file_path in json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
                for model_name, df_data in data.items():
                    self.model_results[model_name] = pd.DataFrame(df_data)

    def plot_auc_curves(self):
        plt.figure(figsize=(10, 8))

        for model_name, df in self.model_results.items():
            fpr, tpr, _ = roc_curve(df['y_test'], df['probabilities'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def calculate_metrics(self, threshold):
        metrics = []

        for model_name, df in self.model_results.items():
            y_true = df['y_test']
            y_pred = df['probabilities'] >= threshold
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            metrics.append({
                'model': model_name,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

        return pd.DataFrame(metrics)

    
    # def calculate_metrics(self):
    #     # Evaluation Metrics
    #     logging.info(f'classification_report: \n{classification_report(self.y_test, self.binary_predictions)}')
    #     logging.info(f'precision_score: {precision_score(self.y_test, self.binary_predictions)}')
    #     logging.info(f'recall_score: {recall_score(self.y_test, self.binary_predictions)}')
    #     logging.info(f'f1_score: {f1_score(self.y_test, self.binary_predictions)}')


    # def plot_ROC_Curve_calibration_curve(self):
    #     # Calculate AUROC
    #     roc_auc, ci_lower, ci_upper = self.compute_auc_confidence_interval(self.y_test, self.probabilities)
    #     logging.info(f'calculate AUROC of the model: {roc_auc}')
    #     fpr, tpr, thresholds = roc_curve(self.y_test, self.probabilities)

    #     plt.figure(figsize=(10, 10))

    #     # Subplot 1 for ROC Curve
    #     plt.subplot(1, 2, 1)
    #     plt.plot([0, 1], [0, 1], linestyle='--')
    #     plt.plot(fpr, tpr, label='AUROC = %0.3f (%0.3f - %0.3f)' % (roc_auc, ci_lower, ci_upper))
    #     plt.xlabel('1-Specificity')
    #     plt.ylabel('Sensitivity')
    #     plt.title('ROC Curve')
    #     plt.legend(handlelength=0)
    #     # Subplot 2 for Calibration Curve# Compute the calibration curve
    #     fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, self.probabilities, n_bins=10)

    #     # Subplot 2 for Calibration Plot
    #     plt.subplot(1, 2, 2)
    #     plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    #     plt.plot([0, 1], [0, 1], "k:")
    #     plt.xlabel('Predicted')
    #     plt.ylabel('Observed')
    #     plt.title('Calibration Plot')
    #     plt.legend(handlelength=0)

    #     # Adjust layout
    #     plt.tight_layout()
    #     plt.show()
    
    



    
    