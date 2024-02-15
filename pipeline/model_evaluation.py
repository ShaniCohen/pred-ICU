import logging
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import shap
import pickle
import datetime
from data_handler import DataHandler
from preprocessing import Preprocessing
import matplotlib.pyplot as plt


class ModelEvaluation:
<<<<<<< HEAD
    def __init__(self, json_files, cutoffs,model_for_shap, data_handler : DataHandler,preprocessing : Preprocessing):
=======
    def __init__(self, json_files):
>>>>>>> 6e90e95e44f863cd617930bf0f3275a8bbaefae2
        # Load all dataframes from json files
        self.model_results = {}
        self.model_params = {}
        for file_path in json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
<<<<<<< HEAD
                for model_name, df_data in data.items():
                    self.model_results[model_name] = pd.DataFrame(df_data)
                    self.model_names.append(model_name.split('_')[0])
                    self.model_params.append(model_name.split('_')[1])
        self.cutoffs = cutoffs
        self.model_shap = model_for_shap
        self.data_handler = data_handler
        self.preprocessing = preprocessing
    # def plot_auc_curves(self):
    #     plt.figure(figsize=(10, 8))
    #
    #     for model_name, df in self.model_results.items():
    #         fpr, tpr, _ = roc_curve(df['y_test'], df['probabilities'])
    #         roc_auc = auc(fpr, tpr)
    #         plt.plot(fpr, tpr, label=f'{model_name.split("_")[0]} (area = {roc_auc:.2f})')
    #
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc='lower right')
    #     plt.show()
    #
    # def calculate_metrics_with_bootstrap(self, list_threshold, n_bootstrap):
    #     metrics = []
    #     bootstrap_cis = {}
    #
    #     for model_name, df in self.model_results.items():
    #         y_true = df['y_test']
    #         probabilities = df['probabilities']
    #         for threshold in list_threshold:
    #             # Calculate metrics for the original sample
    #             y_pred = probabilities >= threshold
    #             precision = precision_score(y_true, y_pred)
    #             recall = recall_score(y_true, y_pred)
    #             f1 = f1_score(y_true, y_pred)
    #
    #             # Bootstrap for CI calculation
    #             bootstrap_metrics = {
    #                 'precision': [],
    #                 'recall': [],
    #                 'f1_score': []
    #             }
    #             for _ in range(n_bootstrap):
    #                 # Generate bootstrap sample
    #                 boot_indices = resample(range(len(y_true)), replace=True)
    #                 y_true_boot = y_true[boot_indices]
    #                 y_pred_boot = probabilities[boot_indices] >= threshold
    #
    #                 # Calculate metrics for the bootstrap sample
    #                 bootstrap_metrics['precision'].append(precision_score(y_true_boot, y_pred_boot))
    #                 bootstrap_metrics['recall'].append(recall_score(y_true_boot, y_pred_boot))
    #                 bootstrap_metrics['f1_score'].append(f1_score(y_true_boot, y_pred_boot))
    #
    #             # Calculate the percentiles for CIs
    #             for metric in bootstrap_metrics:
    #                 lower_bound = np.percentile(bootstrap_metrics[metric], 2.5)
    #                 upper_bound = np.percentile(bootstrap_metrics[metric], 97.5)
    #                 bootstrap_cis[(model_name, threshold, metric)] = (lower_bound, upper_bound)
    #
    #             metrics.append({
    #                 'model': model_name.split('_')[0],
    #                 'threshold': threshold,
    #                 'precision': precision,
    #                 'precision_ci_lower': bootstrap_cis[(model_name, threshold, 'precision')][0],
    #                 'precision_ci_upper': bootstrap_cis[(model_name, threshold, 'precision')][1],
    #                 'recall': recall,
    #                 'recall_ci_lower': bootstrap_cis[(model_name, threshold, 'recall')][0],
    #                 'recall_ci_upper': bootstrap_cis[(model_name, threshold, 'recall')][1],
    #                 'f1_score': f1,
    #                 'f1_score_ci_lower': bootstrap_cis[(model_name, threshold, 'f1_score')][0],
    #                 'f1_score_ci_upper': bootstrap_cis[(model_name, threshold, 'f1_score')][1],
    #             })
    #
    #     metrics_df = pd.DataFrame(metrics)
    #     # Reordering columns for better readability
    #     cols_order = ['model', 'threshold', 'precision', 'precision_ci_lower', 'precision_ci_upper',
    #                   'recall', 'recall_ci_lower', 'recall_ci_upper',
    #                   'f1_score', 'f1_score_ci_lower', 'f1_score_ci_upper']
    #     return metrics_df[cols_order]
    #
    # def plot_precision_recall_curves(self):
    #     plt.figure(figsize=(10, 8))
    #
    #     # Keep track of used positions to avoid overlapping annotations
    #     used_positions = []
    #
    #     for model_name, df in self.model_results.items():
    #         precision, recall, thresholds = precision_recall_curve(df['y_test'], df['probabilities'])
    #         plt.plot(recall, precision, label=f'{model_name.split("_")[0]}')
    #
    #         # Select a few thresholds to annotate on the plot
    #         thresholds_to_annotate = [0.2, 0.4, 0.6, 0.8, 0.9]
    #         for threshold in thresholds_to_annotate:
    #             # Find the closest threshold index
    #             closest_threshold_index = np.argmin(np.abs(thresholds - threshold))
    #             recall_pos = recall[closest_threshold_index + 1]
    #             precision_pos = precision[closest_threshold_index + 1]
    #
    #             # Check if we have already used this position for annotation
    #             if (recall_pos, precision_pos) in used_positions:
    #                 continue  # Skip this one to avoid overlap
    #             used_positions.append((recall_pos, precision_pos))
    #
    #             # Annotate the plot with text and an arrow
    #             plt.annotate(
    #                 f'{threshold:.2f}',
    #                 xy=(recall_pos, precision_pos), xycoords='data',
    #                 xytext=(40, -40), textcoords='offset points',
    #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='black'),
    #                 ha='center', va='center', fontsize=9, color='black',
    #                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
    #             )
    #
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Precision-Recall Curve')
    #     plt.legend(loc='lower left')
    #     plt.grid(True)  # Add a grid for easier readability
    #     plt.show()
    #
    # def melt_df_for_plotting(df, value_vars, ci_vars, id_vars=['model', 'threshold']):
    #     df_melted_values = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
    #                                var_name='Metric', value_name='Value')
    #     df_melted_cis = pd.melt(df, id_vars=id_vars, value_vars=ci_vars,
    #                             var_name='Metric', value_name='CI_Value')
    #
    #     # Combine both DataFrames and calculate the CI size
    #     df_melted = pd.merge(
    #         df_melted_values,
    #         df_melted_cis,
    #         on=id_vars + ['Metric'],
    #         how='left'
    #     )
    #     df_melted['Metric'] = df_melted['Metric'].str.replace('_ci_lower', '').str.replace('_ci_upper', '')
    #     df_melted['CI'] = df_melted.apply(lambda row: row['CI_Value'] - row['Value']
    #     if 'upper' in row['Metric'] else
    #     row['Value'] - row['CI_Value'], axis=1)
    #     return df_melted
    #
    # # Function to plot metrics with confidence intervals using Seaborn
    # def plot_metrics_with_ci(df):
    #     value_vars = ['precision', 'recall', 'f1_score']
    #     ci_lower_vars = [var + '_ci_lower' for var in value_vars]
    #     ci_upper_vars = [var + '_ci_upper' for var in value_vars]
    #
    #     # Prepare the DataFrame for plotting
    #     df_melted = ModelEvaluation.melt_df_for_plotting(df, value_vars, ci_lower_vars + ci_upper_vars)
    #
    #     # Use Seaborn's FacetGrid for plotting
    #     g = sns.FacetGrid(df_melted, col="Metric", hue="model", sharey=False, height=5, aspect=1.2)
    #     g.map(plt.errorbar, "threshold", "Value", "CI", fmt='')
    #
    #     # Adjust the aesthetics
    #     g.add_legend()
    #     g.set_axis_labels("Threshold", "Score")
    #     plt.subplots_adjust(top=0.9)
    #     g.fig.suptitle('Classification Metrics with Confidence Intervals')
    #
    #     plt.show()
    #
    # # Call the function with your dataframe
    #
    # # def calculate_metrics(self):
    # #     # Evaluation Metrics
    # #     logging.info(f'classification_report: \n{classification_report(self.y_test, self.binary_predictions)}')
    # #     logging.info(f'precision_score: {precision_score(self.y_test, self.binary_predictions)}')
    # #     logging.info(f'recall_score: {recall_score(self.y_test, self.binary_predictions)}')
    # #     logging.info(f'f1_score: {f1_score(self.y_test, self.binary_predictions)}')
    #
    # # def plot_ROC_Curve_calibration_curve(self):
    # #     # Calculate AUROC
    # #     roc_auc, ci_lower, ci_upper = self.compute_auc_confidence_interval(self.y_test, self.probabilities)
    # #     logging.info(f'calculate AUROC of the model: {roc_auc}')
    # #     fpr, tpr, thresholds = roc_curve(self.y_test, self.probabilities)
    #
    # #     plt.figure(figsize=(10, 10))
    #
    # #     # Subplot 1 for ROC Curve
    # #     plt.subplot(1, 2, 1)
    # #     plt.plot([0, 1], [0, 1], linestyle='--')
    # #     plt.plot(fpr, tpr, label='AUROC = %0.3f (%0.3f - %0.3f)' % (roc_auc, ci_lower, ci_upper))
    # #     plt.xlabel('1-Specificity')
    # #     plt.ylabel('Sensitivity')
    # #     plt.title('ROC Curve')
    # #     plt.legend(handlelength=0)
    # #     # Subplot 2 for Calibration Curve# Compute the calibration curve
    # #     fraction_of_positives, mean_predicted_value = calibration_curve(self.y_test, self.probabilities, n_bins=10)
    #
    # #     # Subplot 2 for Calibration Plot
    # #     plt.subplot(1, 2, 2)
    # #     plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    # #     plt.plot([0, 1], [0, 1], "k:")
    # #     plt.xlabel('Predicted')
    # #     plt.ylabel('Observed')
    # #     plt.title('Calibration Plot')
    # #     plt.legend(handlelength=0)
    #
    # #     # Adjust layout
    # #     plt.tight_layout()
    # #     plt.show()
=======
                for model_full_name, df_data in data.items():
                    model_short_name = model_full_name.split('_')[0]
                    self.model_results[model_short_name] = pd.DataFrame(df_data)
                    self.model_params[model_short_name] = model_full_name.split('_')[1]
        self.model_names_to_colors = {'LogisticRegression': '#1f77b4', 'XGBClassifier': '#ff7f0e', 'RandomForestClassifier': '#2ca02c', 'Apache': 'black'}
        self.cutoffs_to_colors = {0.01: 'magenta', 0.05: 'lime', 0.1: 'aqua'}
        np.random.seed(42)

    def get_apache_predictions(self):
        training_data_file_path = os.path.abspath('..\\data\\training_v2.csv')
        training_df = pd.read_csv(training_data_file_path)
        apache_predictions = training_df[['apache_4a_hospital_death_prob', 'hospital_death']].copy()
        apache_predictions.rename(columns={'apache_4a_hospital_death_prob': 'probabilities', 'hospital_death': 'y_test'}, inplace=True)
        return apache_predictions

    def plot_roc_curves(self, including_apache=False, including_cutoffs=False, including_confidence_intervals=False):
        cutoffs_to_colors = self.cutoffs_to_colors if including_cutoffs else {}
>>>>>>> 6e90e95e44f863cd617930bf0f3275a8bbaefae2

        fig, ax = plt.subplots()
        ax.set_title('Receiver Operating Characteristic Curve')
        ax.set_xlabel('1 - Specificity (False Positive Rate)')
        ax.set_ylabel('Sensitivity (True Positive Rate)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(which='major', alpha=0.15)
        ax.minorticks_on()
        ax.plot([0, 1], [0, 1], '--', color='grey')

        if including_apache:
            apache_predictions = self.get_apache_predictions()
            invalid_indices = apache_predictions[apache_predictions['probabilities'].isnull() | (apache_predictions['probabilities'] == -1)].index
            for model_name in self.model_results:
                self.model_results[model_name] = self.model_results[model_name].drop(invalid_indices)
            cleaned_apache_predictions = apache_predictions.drop(invalid_indices)
            self.model_results['Apache'] = cleaned_apache_predictions

        mean_fpr = np.linspace(0, 1, 100)

        cutoff_points_positions = {cutoff: [] for cutoff in cutoffs_to_colors.keys()}
        for model_name, df in self.model_results.items():
            y_test = df['y_test']
            probabilities = df['probabilities']

            # Bootstrap
            n_bootstraps = 1000
            bootstrapped_aucs = []
            bootstrapped_sens_spec = {cutoff: {'sensitivity': [], 'specificity': []} for cutoff in cutoffs_to_colors.keys()}
            bootstrapped_tprs = []

            for i in range(n_bootstraps):
                # Sample with replacement
                indices = np.random.choice(np.arange(len(y_test)), size=len(y_test), replace=True)
                y_test_boot = y_test.iloc[indices]
                probabilities_boot = probabilities.iloc[indices]

                fpr, tpr, thresholds = roc_curve(y_test_boot, probabilities_boot)
                roc_auc = auc(fpr, tpr)
                bootstrapped_aucs.append(roc_auc)
                # Interpolate the TPR at the common FPRs
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0 # Ensure starting at 0
                bootstrapped_tprs.append(interp_tpr)

                for cutoff in cutoffs_to_colors.keys():
                    idx = np.argmin(np.abs(thresholds - cutoff))
                    bootstrapped_sens_spec[cutoff]['sensitivity'].append(tpr[idx])
                    bootstrapped_sens_spec[cutoff]['specificity'].append(1 - fpr[idx])

            # Calculate confidence interval for AUC
            sorted_aucs = np.array(bootstrapped_aucs)
            sorted_aucs.sort()
            lower_bound_auc = np.percentile(sorted_aucs, 2.5)
            upper_bound_auc = np.percentile(sorted_aucs, 97.5)

            # Original ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, probabilities)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUROC = {roc_auc:.3f}, CI: {lower_bound_auc:.3f}-{upper_bound_auc:.3f})', color=self.model_names_to_colors[model_name])

            if including_confidence_intervals:
                # Calculate confidence intervals for each FPR
                bootstrapped_tprs = np.array(bootstrapped_tprs)
                tpr_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
                tpr_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
                ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=self.model_names_to_colors[model_name], alpha=0.25)

            for cutoff in cutoffs_to_colors.keys():
                closest_threshold_idx = np.argmin(np.abs(thresholds - cutoff))
                closest_fpr = fpr[closest_threshold_idx]
                closest_tpr = tpr[closest_threshold_idx]

                # Calculate confidence intervals for sensitivity and specificity
                sens_conf_int = np.percentile(bootstrapped_sens_spec[cutoff]['sensitivity'], [2.5, 97.5])
                spec_conf_int = np.percentile(bootstrapped_sens_spec[cutoff]['specificity'], [2.5, 97.5])

                logging.info(f'{model_name}, {cutoff:.0%} cutoff: Sensitivity = {closest_tpr * 100:.2f}%, CI: {sens_conf_int[0] * 100:.2f}-{sens_conf_int[1] * 100:.2f}%, Specificity = {(1 - closest_fpr) * 100:.2f}%, CI: {spec_conf_int[0] * 100:.2f}-{spec_conf_int[1] * 100:.2f}%')

                ax.scatter(closest_fpr, closest_tpr, color=cutoffs_to_colors[cutoff], zorder=5, s=25)
                cutoff_points_positions[cutoff].append((closest_fpr, closest_tpr))

        for cutoff in cutoffs_to_colors.keys():
            average_fpr = np.mean([point[0] for point in cutoff_points_positions[cutoff]])
            average_tpr = np.mean([point[1] for point in cutoff_points_positions[cutoff]])
            ax.annotate(f'{cutoff:.0%} cutoff', xy=(average_fpr, average_tpr - 0.15))
            for point in cutoff_points_positions[cutoff]:
                ax.add_artist(ConnectionPatch((average_fpr + 0.065, average_tpr - 0.125), point, "data", "data", arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=10, fc="w"))

        fig.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        image_name = '_'.join(self.model_results.keys()) + '_roc_with' + ('out' if not including_cutoffs else '') + '_cutoffs_with' + ('out' if not including_confidence_intervals else '') + '_ci' + '.png'
        plt.savefig(image_name, dpi=300)
        plt.close(fig)

    def plot_precision_recall_curves(self, including_apache=False, including_cutoffs=False, including_confidence_intervals=False):
        cutoffs_to_colors = self.cutoffs_to_colors if including_cutoffs else {}

        fig, ax = plt.subplots()
        ax.set_title('Precision-Recall Curve')
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (Positive Predictive Value)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(which='major', alpha=0.15)
        ax.minorticks_on()

        if including_apache:
            apache_predictions = self.get_apache_predictions()
            invalid_indices = apache_predictions[apache_predictions['probabilities'].isnull() | (apache_predictions['probabilities'] == -1)].index
            for model_name in self.model_results:
                self.model_results[model_name] = self.model_results[model_name].drop(invalid_indices)
            cleaned_apache_predictions = apache_predictions.drop(invalid_indices)
            self.model_results['Apache'] = cleaned_apache_predictions

        cutoff_points_positions = {cutoff: [] for cutoff in cutoffs_to_colors.keys()}
        cutoff_ppv_cis = {cutoff: [] for cutoff in cutoffs_to_colors.keys()}  # To store PPV CIs for each cutoff

        for model_name, df in self.model_results.items():
            y_test = df['y_test']
            probabilities = df['probabilities']

            # Bootstrap
            n_bootstraps = 1000
            bootstrapped_scores = []
            bootstrapped_praucs = []
            bootstrapped_ppvs = {cutoff: [] for cutoff in cutoffs_to_colors.keys()}  # To store PPVs for each bootstrap iteration

            for i in range(n_bootstraps):
                # Bootstrap by sampling with replacement on the indices
                indices = np.random.choice(np.arange(len(y_test)), size=len(y_test), replace=True)
                y_true_boot = y_test.iloc[indices]
                y_pred_boot = probabilities.iloc[indices]

                precision, recall, thresholds = precision_recall_curve(y_true_boot, y_pred_boot)
                bootstrapped_scores.append((precision, recall))
                prauc = auc(recall, precision)
                bootstrapped_praucs.append(prauc)

                # Calculate PPVs for each cutoff in this bootstrap iteration
                for cutoff in cutoffs_to_colors.keys():
                    idx = np.where(thresholds > cutoff)[0][0] if np.where(thresholds > cutoff)[0].size > 0 else -1
                    if idx != -1:
                        bootstrapped_ppvs[cutoff].append(precision[idx - 1])

            if including_confidence_intervals:
                # Calculate confidence intervals at each recall level
                recall_levels = np.linspace(0, 1, 100)
                precision_cis = np.zeros((len(recall_levels), 2))
                for i, recall_level in enumerate(recall_levels):
                    precision_at_recall = [np.interp(recall_level, recall[::-1], precision[::-1]) for precision, recall in bootstrapped_scores]
                    lower_bound = np.percentile(precision_at_recall, 2.5)
                    upper_bound = np.percentile(precision_at_recall, 97.5)
                    precision_cis[i] = [lower_bound, upper_bound]

                # Fill between for confidence intervals
                ax.fill_between(recall_levels, precision_cis[:, 0], precision_cis[:, 1], color=self.model_names_to_colors[model_name], alpha=0.25)

            # Calculate and log PPV confidence intervals for each cutoff
            for cutoff in cutoffs_to_colors.keys():
                ppv_lower_bound = np.percentile(bootstrapped_ppvs[cutoff], 2.5) if bootstrapped_ppvs[cutoff] else 0
                ppv_upper_bound = np.percentile(bootstrapped_ppvs[cutoff], 97.5) if bootstrapped_ppvs[cutoff] else 0
                cutoff_ppv_cis[cutoff] = (ppv_lower_bound, ppv_upper_bound)
                logging.info(f'{model_name}, {cutoff:.0%} cutoff: PPV = {np.mean(bootstrapped_ppvs[cutoff]) * 100:.2f}, CI: {ppv_lower_bound * 100:.2f}-{ppv_upper_bound * 100:.2f}%')

            # Calculate confidence intervals for PRAUC
            prauc_lower_bound = np.percentile(bootstrapped_praucs, 2.5)
            prauc_upper_bound = np.percentile(bootstrapped_praucs, 97.5)

            # Plotting
            precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
            prauc = auc(recall, precision)
            ax.plot(recall, precision, label=f'{model_name} (PRAUC = {prauc:.3f}, CI: {prauc_lower_bound:.3f}-{prauc_upper_bound:.3f})', color=self.model_names_to_colors[model_name])

            for cutoff in cutoffs_to_colors.keys():
                idx = np.where(thresholds > cutoff)[0][0] if np.where(thresholds > cutoff)[0].size > 0 else -1
                if idx != -1:
                    cutoff_precision = precision[idx - 1]
                    cutoff_recall = recall[idx - 1]
                    ax.scatter(cutoff_recall, cutoff_precision, color=cutoffs_to_colors[cutoff], zorder=5, s=25)
                    cutoff_points_positions[cutoff].append((cutoff_recall, cutoff_precision))

        for cutoff in cutoffs_to_colors.keys():
            average_recall = np.mean([point[0] for point in cutoff_points_positions[cutoff]])
            average_precision = np.mean([point[1] for point in cutoff_points_positions[cutoff]])
            temp = 0.02 if cutoff == 0.1 else 0
            ax.annotate(f'{cutoff:.0%} cutoff', xy=(average_recall - 0.245 - temp, average_precision - 0.012))
            for point in cutoff_points_positions[cutoff]:
                ax.add_artist(ConnectionPatch((average_recall - 0.1, average_precision), point, "data", "data", arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=10, fc="w"))

        fig.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        image_name = '_'.join([model_name for model_name in self.model_results.keys()]) + '_precision_recall_with' + ('out' if not including_cutoffs else '') + '_cutoffs_with' + ('out' if not including_confidence_intervals else '') + '_ci' '.png'
        plt.savefig(image_name, dpi=300)
        plt.close(fig)

    def plot_sensitivity_percent_positives_curves(self, including_apache=False, including_cutoffs=False, including_confidence_intervals=False):
        cutoffs_to_colors = self.cutoffs_to_colors if including_cutoffs else {}

        fig, ax = plt.subplots()
        ax.set_title('Sensitivity as a Function of Percent Positives')
        ax.set_xlabel('Percent Positives')
        ax.set_ylabel('Sensitivity')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(which='major', alpha=0.15)
        ax.minorticks_on()

        if including_apache:
            apache_predictions = self.get_apache_predictions()
            invalid_indices = apache_predictions[apache_predictions['probabilities'].isnull() | (apache_predictions['probabilities'] == -1)].index
            for model_name in self.model_results:
                self.model_results[model_name] = self.model_results[model_name].drop(invalid_indices)
            cleaned_apache_predictions = apache_predictions.drop(invalid_indices)
            self.model_results['Apache'] = cleaned_apache_predictions

        cutoff_points_positions = {cutoff: [] for cutoff in cutoffs_to_colors.keys()}
        for model_name, df in self.model_results.items():
            y_test = df['y_test']
            probabilities = df['probabilities']

            fpr, tpr, thresholds = roc_curve(y_test, probabilities)
            P = np.sum(y_test == 1)
            N = np.sum(y_test == 0)
            FP = fpr * N
            TP = tpr * P
            percent_positives = (TP + FP) / (P + N)
            lift = np.divide(tpr, percent_positives, where=(percent_positives > 0))

            # Plot the sensitivity-percent positives curve
            ax.plot(percent_positives, tpr, label=model_name, color=self.model_names_to_colors[model_name])

            if including_confidence_intervals:
                n_bootstraps = 1000

                common_thresholds = np.linspace(0, 1, 100)

                bootstrapped_sensitivities = []
                bootstrapped_percent_positives_lifts = {cutoff: {'percent_positive': [], 'lift': []} for cutoff in cutoffs_to_colors.keys()}

                for i in range(n_bootstraps):
                    indices = np.random.choice(np.arange(len(y_test)), size=len(y_test), replace=True)
                    if len(np.unique(y_test.iloc[indices])) < 2:
                        continue # If sample doesn't include both classes, skip this iteration

                    # Recalculate ROC curve
                    fpr_boot, tpr_boot, thresholds_boot = roc_curve(y_test.iloc[indices], probabilities.iloc[indices])
                    P_boot = np.sum(y_test.iloc[indices] == 1)
                    N_boot = np.sum(y_test.iloc[indices] == 0)
                    FP_boot = fpr_boot * N_boot
                    TP_boot = tpr_boot * P_boot
                    percent_positives_boot = (TP_boot + FP_boot) / (P_boot + N_boot) if (P_boot + N_boot) > 0 else 0
                    lift_boot = np.divide(tpr_boot, percent_positives_boot, where=(percent_positives_boot > 0))

                    # Interpolate TPRs at common percent positive rates or thresholds
                    # This ensures that each bootstrap sample contributes a TPR value for each common threshold
                    interpolated_sensitivities = np.interp(common_thresholds, percent_positives_boot, tpr_boot)
                    bootstrapped_sensitivities.append(interpolated_sensitivities)

                    for cutoff in cutoffs_to_colors.keys():
                        idx = (np.abs(thresholds_boot - cutoff)).argmin()
                        bootstrapped_percent_positives_lifts[cutoff]['percent_positive'].append(percent_positives_boot[idx])
                        bootstrapped_percent_positives_lifts[cutoff]['lift'].append(lift_boot[idx])

                bootstrapped_sensitivities = np.array(bootstrapped_sensitivities)
                lower_bounds_sensitivity, upper_bounds_sensitivity = np.percentile(bootstrapped_sensitivities, [2.5, 97.5], axis=0)
                ax.fill_between(common_thresholds, lower_bounds_sensitivity, upper_bounds_sensitivity, color=self.model_names_to_colors[model_name], alpha=0.25)

                for cutoff, color in cutoffs_to_colors.items():
                    idx = (np.abs(thresholds - cutoff)).argmin()

                    cutoff_percent_positive = percent_positives[idx]
                    cutoff_lift = lift[idx]

                    percent_positives_ci_lower, percent_positives_ci_upper = np.percentile(bootstrapped_percent_positives_lifts[cutoff]['percent_positive'], [2.5, 97.5])
                    lift_ci_lower, lift_ci_upper = np.percentile(bootstrapped_percent_positives_lifts[cutoff]['lift'], [2.5, 97.5])

                    # Log the metrics including their confidence intervals for every cutoff
                    logging.info(f'{model_name}, {cutoff:.0%} cutoff: Percent Positive = {cutoff_percent_positive * 100:.2f}%, CI: {percent_positives_ci_lower * 100:.2f}-{percent_positives_ci_upper * 100:.2f}%, Lift = {cutoff_lift:.2f}, CI: {lift_ci_lower:.2f}-{lift_ci_upper:.2f}')

                    ax.scatter(percent_positives[idx], tpr[idx], color=cutoffs_to_colors[cutoff], zorder=5, s=25)
                    cutoff_points_positions[cutoff].append((percent_positives[idx], tpr[idx]))

        for cutoff in cutoffs_to_colors.keys():
            average_percent_positive = np.mean([point[0] for point in cutoff_points_positions[cutoff]])
            average_sensitivity = np.mean([point[1] for point in cutoff_points_positions[cutoff]])
            ax.annotate(f'{cutoff:.0%} cutoff', xy=(average_percent_positive + 0.03, average_sensitivity - 0.15))
            for point in cutoff_points_positions[cutoff]:
                ax.add_artist(ConnectionPatch((average_percent_positive + 0.085, average_sensitivity - 0.12), point, "data", "data", arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=10, fc="w"))

        fig.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        image_name = '_'.join(self.model_results.keys()) + '_sensitivity_percent_positives_with' + ('out' if not including_cutoffs else '') + '_cutoffs_with' + ('out' if not including_confidence_intervals else '') + '_ci' + '.png'
        plt.savefig(image_name, dpi=300)
        plt.close(fig)

    def plot_precision_percent_positives_curves(self, including_apache=False, including_cutoffs=False, including_confidence_intervals=False):
        cutoffs_to_colors = self.cutoffs_to_colors if including_cutoffs else {}

        fig, ax = plt.subplots()
        ax.set_title('Precision as a Function of Percent Positives')
        ax.set_xlabel('Percent Positives')
        ax.set_ylabel('Precision (Positive Predictive Value)')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.grid(which='major', alpha=0.15)
        ax.minorticks_on()

        if including_apache:
            apache_predictions = self.get_apache_predictions()
            invalid_indices = apache_predictions[apache_predictions['probabilities'].isnull() | (apache_predictions['probabilities'] == -1)].index
            for model_name in self.model_results:
                self.model_results[model_name] = self.model_results[model_name].drop(invalid_indices)
            cleaned_apache_predictions = apache_predictions.drop(invalid_indices)
            self.model_results['Apache'] = cleaned_apache_predictions

        cutoff_points_positions = {cutoff: [] for cutoff in cutoffs_to_colors.keys()}
        for model_name, df in self.model_results.items():
            y_test = df['y_test']
            probabilities = df['probabilities']

            fpr, tpr, thresholds = roc_curve(y_test, probabilities)
            P = np.sum(y_test == 1)
            N = np.sum(y_test == 0)
            FP = fpr * N
            TP = tpr * P

            precision = np.divide(TP, (TP + FP), where=((TP + FP) > 0))
            percent_positives = np.divide((TP + FP), (P + N), where=(P + N) > 0)

            # Plot the precision-percent positives curve
            ax.plot(percent_positives, precision, label=model_name, color=self.model_names_to_colors[model_name])

            if including_confidence_intervals:
                n_bootstraps = 1000
                common_thresholds = np.linspace(0, 1, 100)
                bootstrapped_precisions = []

                for i in range(n_bootstraps):
                    # Bootstrap by sampling with replacement on the indices
                    indices = np.random.choice(np.arange(len(y_test)), size=len(y_test), replace=True)
                    if len(np.unique(y_test.iloc[indices])) < 2:
                        continue  # If sample doesn't include both classes, skip this iteration

                    fpr_boot, tpr_boot, thresholds_boot = roc_curve(y_test.iloc[indices], probabilities.iloc[indices])
                    P_boot = np.sum(y_test.iloc[indices] == 1)
                    N_boot = np.sum(y_test.iloc[indices] == 0)
                    FP_boot = fpr_boot * N_boot
                    TP_boot = tpr_boot * P_boot
                    precision_boot = np.divide(TP_boot, (TP_boot + FP_boot), where=((TP_boot + FP_boot) > 0))
                    percent_positives_boot = np.divide((TP_boot + FP_boot), (P_boot + N_boot), where=(P_boot + N_boot) > 0)

                    #Interpolate precision at common thresholds
                    interpolated_precisions = np.interp(common_thresholds, percent_positives_boot, precision_boot)
                    bootstrapped_precisions.append(interpolated_precisions)

                # Calculate the percentile for the lower and upper bound
                lower_bounds_precision, upper_bounds_precision = np.percentile(bootstrapped_precisions, [2.5, 97.5], axis=0)
                # Fill between for confidence intervals
                ax.fill_between(common_thresholds, lower_bounds_precision, upper_bounds_precision, color=self.model_names_to_colors[model_name], alpha=0.25)


            for cutoff in cutoffs_to_colors.keys():
                # Find the closest threshold index
                idx = (np.abs(thresholds - cutoff)).argmin()
                cutoff_precision = precision[idx]
                cutoff_percent_positive = percent_positives[idx]

                ax.scatter(cutoff_percent_positive, cutoff_precision, color=cutoffs_to_colors[cutoff], zorder=5, s=25)
                cutoff_points_positions[cutoff].append((cutoff_percent_positive, cutoff_precision))

        for cutoff in cutoffs_to_colors.keys():
            temp = 0.035 if cutoff == 0.1 else 0.0175 if cutoff == 0.05 else 0
            temp2 = 0.0135 if cutoff == 0.1 else 0
            average_percent_positive = np.mean([point[0] for point in cutoff_points_positions[cutoff]])
            average_precision = np.mean([point[1] for point in cutoff_points_positions[cutoff]])
            ax.annotate(f'{cutoff:.0%} cutoff', xy=(average_percent_positive - 0.06 - temp2, average_precision + 0.17 + temp))
            for point in cutoff_points_positions[cutoff]:
                ax.add_artist(ConnectionPatch((average_percent_positive, average_precision + 0.17 + temp), point, "data", "data", arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=10, fc="w"))

        fig.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        image_name = '_'.join(self.model_results.keys()) + '_precision_percent_positives_with' + ('out' if not including_cutoffs else '') + '_cutoffs_with' + ('out' if not including_confidence_intervals else '') + '_ci' + '.png'
        plt.savefig(image_name, dpi=300)
        plt.close(fig)

    def generate_predictions_files(self):
        for model_name, df in self.model_results.items():
            df.drop(columns=['binary_predictions'], inplace=True)
            df = df[['probabilities', 'y_test']]
            df.rename(columns={'y_test': 'labels', 'probabilities': 'predictions'}, inplace=True)
            predictions_file_path = os.path.abspath(f'.\\calibration\\{model_name}_predictions.csv')
            df.to_csv(predictions_file_path, index=False)
            



# ------------------------------------------------------------------------------------------
#----------------------------------- shap --------------------------------------------------
    def plot_shap_waterfall(base_value, shap_values, feature_names, actual_prediction):
        """
        Plots a SHAP waterfall chart for a single prediction.
        
        Parameters:
        - base_value: The base value (average model output over the dataset).
        - shap_values: Array of SHAP values for each feature for a single prediction.
        - feature_names: List of feature names corresponding to the SHAP values.
        - actual_prediction: The actual prediction for the instance.
        """
        # Start with the base value
        start_value = base_value
        # Initialize the cumulative sum of SHAP values
        cum_shap_values = [start_value]
        # Calculate cumulative sum
        for shap_value in shap_values:
            start_value += shap_value
            cum_shap_values.append(start_value)
        
        # Prepare the plotting data
        step_values = [base_value] + list(cum_shap_values[:-1])
        end_values = cum_shap_values
        
        # Plotting
        plt.figure(figsize=(10, 6))
        for i in range(len(shap_values)):
            plt.fill_between([i, i + 1], step_values[i], end_values[i], 
                            color='skyblue' if shap_values[i] >= 0 else 'salmon', step='pre')
        
        # Adding final prediction
        plt.plot([0, len(shap_values)], [actual_prediction, actual_prediction], 'k--', label='Final Prediction')
        
        # Customize the plot
        plt.xticks(ticks=range(len(feature_names) + 1), labels=['Base Value'] + feature_names, rotation=45, ha="right")
        plt.ylabel('Prediction Value')
        plt.title('SHAP Waterfall Plot for a Single Prediction')
        plt.legend()
        plt.grid(True)
        
        # Show plot
        plt.tight_layout()
        plt.show()

    # Example usage (assuming you have the necessary values calculated)
    # base_value = 0.5  # Example base value
    # shap_values = [0.1, -0.2, 0.05, 0.3]  # Example SHAP values for each feature for a single prediction
    # feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4']  # Example feature names
    # actual_prediction = 0.75  # Example actual prediction for the instance

    # plot_shap_waterfall(base_value, shap_values, feature_names, actual_prediction)
    # # Plot the SHAP waterfall chart
    
    
    # fit the model for calculating shap values
    # because we need to keep the interpretability of the model we will not be using scaling
    # also xgboost is a good model because it does not require scaling
    def fit_model_for_shap(self):
        # before we do all the preprocessing, we will use the class of the preprocessing for that
        # use Data handler class to load the data
        main_data = self.data_handler.load_data()
        
        X = main_data.drop(['encounter_id', 'patient_id', 'hospital_death', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'readmission_status'], axis=1)
        y = main_data['hospital_death']
        x_features_list = X.columns.tolist()
        X_train_processed, fitted_scaler, feature_info_dtype, dict_of_fill_values, encoder_info = self.preprocessing.run_preprocessing_fit(data=X,                                                                                                                                 list_of_x_features_for_model=x_features_list,
                                                                                                                                           to_scale=False)
        model = self.model_shap
        model.fit(X_train_processed, y)
        return model, X_train_processed, y


    def calculate_and_save_shap_values(self,model, X):
        """
        Calculates SHAP values for a given model and dataset, then saves the values to a pickle file.
        
        Parameters:
        - model: A trained machine learning model.
        - X: The dataset (features) for which SHAP values are to be calculated (numpy array or pandas DataFrame).
        - path: The file path where the SHAP values should be saved (string).
        """
        # Initialize the SHAP Tree explainer (use KernelExplainer, DeepExplainer, or GradientExplainer as needed for other model types)
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Directory path for JSON file
        base_directory = os.path.dirname(os.path.dirname(DataHandler.file_path))
        shap_directory = os.path.join(base_directory, 'shap_values')
        if not os.path.exists(shap_directory):
            os.makedirs(shap_directory)

        # File path for JSON file
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_file_path = os.path.join(shap_directory, f'shap_values_{date}.pkl')
        
        # Save the SHAP values to a pickle file
        with open(json_file_path, 'wb') as file:
            pickle.dump(shap_values, file)
     
            
    def run_shap(self):
        # Fit the model
        model, X_train_processed, _ = self.fit_model_for_shap()
        # Calculate and save SHAP values
        self.calculate_and_save_shap_values(model, X_train_processed)
        # print(f"SHAP values saved to {path}")
        




    

