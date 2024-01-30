        
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
from sklearn.utils import resample

class ModelEvaluation:
    def __init__(self, json_files):
        # Load all dataframes from json files
        self.model_results = {}
        self.model_names = []
        self.model_params = []
        for file_path in json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
                for model_name, df_data in data.items():
                    self.model_results[model_name] = pd.DataFrame(df_data)
                    self.model_names.append(model_name.split('_')[0])
                    self.model_params.append(model_name.split('_')[1])

    def plot_auc_curves(self):
        plt.figure(figsize=(10, 8))

        for model_name, df in self.model_results.items():
            fpr, tpr, _ = roc_curve(df['y_test'], df['probabilities'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name.split("_")[0]} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()


    def calculate_metrics_with_bootstrap(self, list_threshold, n_bootstrap):
        metrics = []
        bootstrap_cis = {}

        for model_name, df in self.model_results.items():
            y_true = df['y_test']
            probabilities = df['probabilities']
            for threshold in list_threshold:
                # Calculate metrics for the original sample
                y_pred = probabilities >= threshold
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)

                # Bootstrap for CI calculation
                bootstrap_metrics = {
                    'precision': [],
                    'recall': [],
                    'f1_score': []
                }
                for _ in range(n_bootstrap):
                    # Generate bootstrap sample
                    boot_indices = resample(range(len(y_true)), replace=True)
                    y_true_boot = y_true[boot_indices]
                    y_pred_boot = probabilities[boot_indices] >= threshold
                    
                    # Calculate metrics for the bootstrap sample
                    bootstrap_metrics['precision'].append(precision_score(y_true_boot, y_pred_boot))
                    bootstrap_metrics['recall'].append(recall_score(y_true_boot, y_pred_boot))
                    bootstrap_metrics['f1_score'].append(f1_score(y_true_boot, y_pred_boot))

                # Calculate the percentiles for CIs
                for metric in bootstrap_metrics:
                    lower_bound = np.percentile(bootstrap_metrics[metric], 2.5)
                    upper_bound = np.percentile(bootstrap_metrics[metric], 97.5)
                    bootstrap_cis[(model_name, threshold, metric)] = (lower_bound, upper_bound)

                metrics.append({
                    'model': model_name.split('_')[0],
                    'threshold': threshold,
                    'precision': precision,
                    'precision_ci_lower': bootstrap_cis[(model_name, threshold, 'precision')][0],
                    'precision_ci_upper': bootstrap_cis[(model_name, threshold, 'precision')][1],
                    'recall': recall,
                    'recall_ci_lower': bootstrap_cis[(model_name, threshold, 'recall')][0],
                    'recall_ci_upper': bootstrap_cis[(model_name, threshold, 'recall')][1],
                    'f1_score': f1,
                    'f1_score_ci_lower': bootstrap_cis[(model_name, threshold, 'f1_score')][0],
                    'f1_score_ci_upper': bootstrap_cis[(model_name, threshold, 'f1_score')][1],
                })

        metrics_df = pd.DataFrame(metrics)
        # Reordering columns for better readability
        cols_order = ['model', 'threshold', 'precision', 'precision_ci_lower', 'precision_ci_upper', 
                    'recall', 'recall_ci_lower', 'recall_ci_upper', 
                    'f1_score', 'f1_score_ci_lower', 'f1_score_ci_upper']
        return metrics_df[cols_order]




    def plot_precision_recall_curves(self):
        plt.figure(figsize=(10, 8))

        # Keep track of used positions to avoid overlapping annotations
        used_positions = []

        for model_name, df in self.model_results.items():
            precision, recall, thresholds = precision_recall_curve(df['y_test'], df['probabilities'])
            plt.plot(recall, precision, label=f'{model_name.split("_")[0]}')

            # Select a few thresholds to annotate on the plot
            thresholds_to_annotate = [0.2, 0.4, 0.6, 0.8, 0.9]
            for threshold in thresholds_to_annotate:
                # Find the closest threshold index
                closest_threshold_index = np.argmin(np.abs(thresholds - threshold))
                recall_pos = recall[closest_threshold_index + 1]
                precision_pos = precision[closest_threshold_index + 1]

                # Check if we have already used this position for annotation
                if (recall_pos, precision_pos) in used_positions:
                    continue  # Skip this one to avoid overlap
                used_positions.append((recall_pos, precision_pos))

                # Annotate the plot with text and an arrow
                plt.annotate(
                    f'{threshold:.2f}',
                    xy=(recall_pos, precision_pos), xycoords='data',
                    xytext=(40, -40), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='black'),
                    ha='center', va='center', fontsize=9, color='black',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
                )

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)  # Add a grid for easier readability
        plt.show()

        
    def melt_df_for_plotting(df, value_vars, ci_vars, id_vars=['model', 'threshold']):
        df_melted_values = pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
                                var_name='Metric', value_name='Value')
        df_melted_cis = pd.melt(df, id_vars=id_vars, value_vars=ci_vars, 
                                var_name='Metric', value_name='CI_Value')
        
        # Combine both DataFrames and calculate the CI size
        df_melted = pd.merge(
            df_melted_values, 
            df_melted_cis, 
            on=id_vars + ['Metric'],
            how='left'
        )
        df_melted['Metric'] = df_melted['Metric'].str.replace('_ci_lower', '').str.replace('_ci_upper', '')
        df_melted['CI'] = df_melted.apply(lambda row: row['CI_Value'] - row['Value']
                                        if 'upper' in row['Metric'] else
                                        row['Value'] - row['CI_Value'], axis=1)
        return df_melted

    # Function to plot metrics with confidence intervals using Seaborn
    def plot_metrics_with_ci(df):
        value_vars = ['precision', 'recall', 'f1_score']
        ci_lower_vars = [var + '_ci_lower' for var in value_vars]
        ci_upper_vars = [var + '_ci_upper' for var in value_vars]
        
        # Prepare the DataFrame for plotting
        df_melted = ModelEvaluation.melt_df_for_plotting(df, value_vars, ci_lower_vars + ci_upper_vars)

        # Use Seaborn's FacetGrid for plotting
        g = sns.FacetGrid(df_melted, col="Metric", hue="model", sharey=False, height=5, aspect=1.2)
        g.map(plt.errorbar, "threshold", "Value", "CI", fmt='')
        
        # Adjust the aesthetics
        g.add_legend()
        g.set_axis_labels("Threshold", "Score")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle('Classification Metrics with Confidence Intervals')

        plt.show()

    # Call the function with your dataframe

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
    
    



    
    