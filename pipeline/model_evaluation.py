import logging
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve, brier_score_loss
import os
import shap
import pickle
import datetime
from data_handler import DataHandler
from preprocessing import Preprocessing


class ModelEvaluation:
    def __init__(self, json_files):
        # Load all dataframes from json files
        self.model_results = {}
        self.model_params = {}
        for file_path in json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
                for model_full_name, df_data in data.items():
                    model_short_name = model_full_name.split('_')[0]
                    self.model_results[model_short_name] = pd.DataFrame(df_data)
        self.model_names_to_colors = {'LogisticRegression': '#1f77b4', 'XGBClassifier': '#ff7f0e', 'RandomForestClassifier': '#2ca02c', 'Apache': '#000000'}
        self.cutoffs_to_colors = {0.01: 'magenta', 0.05: 'lime', 0.1: 'aqua'}
        # self.cutoffs_to_colors = {0.15: 'magenta', 0.2: 'lime'}
        self.number_of_bootstrap_samples = 1000

    def get_apache_predictions(self):
        training_data_file_path = os.path.abspath('..\\data\\training_v2.csv')
        training_df = pd.read_csv(training_data_file_path)
        apache_predictions = training_df[['apache_4a_hospital_death_prob', 'hospital_death']].copy()
        apache_predictions.rename(columns={'apache_4a_hospital_death_prob': 'probabilities', 'hospital_death': 'y_test'}, inplace=True)
        return apache_predictions

    def plot_roc_curves(self, including_apache=False, including_cutoffs=False, including_confidence_intervals=False, including_bootstrap_histograms=False):
        cutoffs_to_colors = self.cutoffs_to_colors if including_cutoffs else {}
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
            bootstrapped_aucs = []
            bootstrapped_sens_spec = {cutoff: {'sensitivity': [], 'specificity': []} for cutoff in cutoffs_to_colors.keys()}
            bootstrapped_tprs = []

            for i in range(self.number_of_bootstrap_samples):
                # Sample with replacement
                indices = np.random.choice(np.arange(len(y_test)), size=len(y_test), replace=True)
                y_test_boot = y_test.iloc[indices]
                probabilities_boot = probabilities.iloc[indices]

                fpr, tpr, thresholds = roc_curve(y_test_boot, probabilities_boot)
                roc_auc = auc(fpr, tpr)
                bootstrapped_aucs.append(roc_auc)
                # Interpolate the TPR at the common FPRs
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0  # Ensure starting at 0
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

            if including_bootstrap_histograms:
                # Plotting the histogram for bootstrapped AUCs
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(sorted_aucs, bins=12, color=self.model_names_to_colors[model_name], ec="k")
                ax_hist.set_title(f'Distribution of AUROC Scores Using {self.number_of_bootstrap_samples} Bootstraps')
                ax_hist.set_xlabel('AUROC Score')
                ax_hist.set_ylabel('Frequency')
                fig_hist.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
                ax_hist.axvline(x=lower_bound_auc, color='dimgrey', linestyle='--', linewidth=1.25)
                ax_hist.axvline(x=upper_bound_auc, color='dimgrey', linestyle='--', linewidth=1.25)
                legend_patch = Patch(color=self.model_names_to_colors[model_name], label=f'{model_name} (AUROC = {roc_auc:.3f}, CI: {lower_bound_auc:.3f}-{upper_bound_auc:.3f})')
                ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

                # Save the histogram image
                histogram_image_name = model_name + '_auc_histogram.png'
                plt.savefig(histogram_image_name, dpi=300)
                plt.close(fig_hist)

                # Plotting the histograms for bootstrapped sensitivities per cutoff
                for cutoff in cutoffs_to_colors.keys():
                    fig_hist, ax_hist = plt.subplots()
                    ax_hist.hist(bootstrapped_sens_spec[cutoff]['sensitivity'], bins=12, color=cutoffs_to_colors[cutoff], ec="k")
                    ax_hist.set_title(f'Distribution of Sensitivities Using {self.number_of_bootstrap_samples} Bootstraps at {cutoff:.0%} Cutoff')
                    ax_hist.set_xlabel('Sensitivity')
                    ax_hist.set_ylabel('Frequency')
                    fig_hist.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
                    ax_hist.axvline(x=np.percentile(bootstrapped_sens_spec[cutoff]["sensitivity"], 2.5), color='dimgrey', linestyle='--', linewidth=1.25)
                    ax_hist.axvline(x=np.percentile(bootstrapped_sens_spec[cutoff]["sensitivity"], 97.5), color='dimgrey', linestyle='--', linewidth=1.25)
                    legend_patch = Patch(color=cutoffs_to_colors[cutoff], label=f'{model_name} (Sensitivity = {np.mean(bootstrapped_sens_spec[cutoff]["sensitivity"]):.3f}, CI: {np.percentile(bootstrapped_sens_spec[cutoff]["sensitivity"], 2.5):.3f}-{np.percentile(bootstrapped_sens_spec[cutoff]["sensitivity"], 97.5):.3f})')
                    ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

                    # Save the histogram image
                    histogram_image_name = model_name + f'_sensitivity_{cutoff:.0%}_cutoff_histogram.png'
                    plt.savefig(histogram_image_name, dpi=300)
                    plt.close(fig_hist)

                # Plotting the histograms for bootstrapped specificities per cutoff
                for cutoff in cutoffs_to_colors.keys():
                    fig_hist, ax_hist = plt.subplots()
                    ax_hist.hist(bootstrapped_sens_spec[cutoff]['specificity'], bins=12, color=cutoffs_to_colors[cutoff], ec="k")
                    ax_hist.set_title(f'Distribution of Specificities Using {self.number_of_bootstrap_samples} Bootstraps at {cutoff:.0%} Cutoff')
                    ax_hist.set_xlabel('Specificity')
                    ax_hist.set_ylabel('Frequency')
                    fig_hist.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
                    ax_hist.axvline(x=np.percentile(bootstrapped_sens_spec[cutoff]["specificity"], 2.5), color='dimgrey', linestyle='--', linewidth=1.25)
                    ax_hist.axvline(x=np.percentile(bootstrapped_sens_spec[cutoff]["specificity"], 97.5), color='dimgrey', linestyle='--', linewidth=1.25)
                    legend_patch = Patch(color=cutoffs_to_colors[cutoff], label=f'{model_name} (Specificity = {np.mean(bootstrapped_sens_spec[cutoff]["specificity"]):.3f}, CI: {np.percentile(bootstrapped_sens_spec[cutoff]["specificity"], 2.5):.3f}-{np.percentile(bootstrapped_sens_spec[cutoff]["specificity"], 97.5):.3f})')
                    ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

                    # Save the histogram image
                    histogram_image_name = model_name + f'_specificity_{cutoff:.0%}_cutoff_histogram.png'
                    plt.savefig(histogram_image_name, dpi=300)
                    plt.close(fig_hist)

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

    def plot_precision_recall_curves(self, including_apache=False, including_cutoffs=False, including_confidence_intervals=False, including_bootstrap_histograms=False):
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
            bootstrapped_scores = []
            bootstrapped_praucs = []
            bootstrapped_ppvs = {cutoff: [] for cutoff in cutoffs_to_colors.keys()}  # To store PPVs for each bootstrap iteration

            for i in range(self.number_of_bootstrap_samples):
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

            # Calculate confidence intervals for PRAUC
            prauc_lower_bound = np.percentile(bootstrapped_praucs, 2.5)
            prauc_upper_bound = np.percentile(bootstrapped_praucs, 97.5)

            # Plotting
            precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
            prauc = auc(recall, precision)
            ax.plot(recall, precision, label=f'{model_name} (PRAUC = {prauc:.3f}, CI: {prauc_lower_bound:.3f}-{prauc_upper_bound:.3f})', color=self.model_names_to_colors[model_name])

            if including_bootstrap_histograms:
                # Plotting the histogram for bootstrapped PRAUCs
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(bootstrapped_praucs, bins=12, color=self.model_names_to_colors[model_name], ec="k")
                ax_hist.set_title(f'Distribution of PRAUC Scores Using {self.number_of_bootstrap_samples} Bootstraps')
                ax_hist.set_xlabel('PRAUC Score')
                ax_hist.set_ylabel('Frequency')
                fig_hist.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
                ax_hist.axvline(x=prauc_lower_bound, color='dimgrey', linestyle='--', linewidth=1.25)
                ax_hist.axvline(x=prauc_upper_bound, color='dimgrey', linestyle='--', linewidth=1.25)
                legend_patch = Patch(color=self.model_names_to_colors[model_name], label=f'{model_name} (PRAUC = {prauc:.3f}, CI: {prauc_lower_bound:.3f}-{prauc_upper_bound:.3f})')
                ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

                # Save the histogram image
                histogram_image_name = model_name + '_prauc_histogram.png'
                plt.savefig(histogram_image_name, dpi=300)
                plt.close(fig_hist)

                #Plotting the histograms for bootstrapped PPVs per cutoff
                for cutoff in cutoffs_to_colors.keys():
                    fig_hist, ax_hist = plt.subplots()
                    ax_hist.hist(bootstrapped_ppvs[cutoff], bins=12, color=cutoffs_to_colors[cutoff], ec="k")
                    ax_hist.set_title(f'Distribution of PPVs Using {self.number_of_bootstrap_samples} Bootstraps at {cutoff:.0%} Cutoff')
                    ax_hist.set_xlabel('PPV')
                    ax_hist.set_ylabel('Frequency')
                    fig_hist.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
                    ax_hist.axvline(x=np.percentile(bootstrapped_ppvs[cutoff], 2.5), color='dimgrey', linestyle='--', linewidth=1.25)
                    ax_hist.axvline(x=np.percentile(bootstrapped_ppvs[cutoff], 97.5), color='dimgrey', linestyle='--', linewidth=1.25)
                    legend_patch = Patch(color=cutoffs_to_colors[cutoff], label=f'{model_name} (PPV = {np.mean(bootstrapped_ppvs[cutoff]):.3f}, CI: {np.percentile(bootstrapped_ppvs[cutoff], 2.5):.3f}-{np.percentile(bootstrapped_ppvs[cutoff], 97.5):.3f})')
                    ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

                    # Save the histogram image
                    histogram_image_name = model_name + f'_ppv_{cutoff:.0%}_cutoff_histogram.png'
                    plt.savefig(histogram_image_name, dpi=300)
                    plt.close(fig_hist)

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
                logging.info(f'{model_name}, {cutoff:.0%} cutoff: PPV = {np.mean(bootstrapped_ppvs[cutoff]) * 100:.2f}%, CI: {ppv_lower_bound * 100:.2f}-{ppv_upper_bound * 100:.2f}%')

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

    def plot_sensitivity_percent_positives_curves(self, including_apache=False, including_cutoffs=False, including_confidence_intervals=False, including_bootstrap_histograms=False):
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
                common_thresholds = np.linspace(0, 1, 100)

                bootstrapped_sensitivities = []
                bootstrapped_percent_positives_lifts = {cutoff: {'percent_positive': [], 'lift': []} for cutoff in cutoffs_to_colors.keys()}

                for i in range(self.number_of_bootstrap_samples):
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

                if including_bootstrap_histograms:
                    # Plotting the histogram for bootstrapped percent positives per cutoff
                    for cutoff in cutoffs_to_colors.keys():
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.hist(bootstrapped_percent_positives_lifts[cutoff]['percent_positive'], bins=12, color=cutoffs_to_colors[cutoff], ec="k")
                        ax_hist.set_title(f'Distribution of Percent Positives Using {self.number_of_bootstrap_samples} Bootstraps at {cutoff:.0%} Cutoff')
                        ax_hist.set_xlabel('Percent Positives')
                        ax_hist.set_ylabel('Frequency')
                        fig_hist.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
                        ax_hist.axvline(x=np.percentile(bootstrapped_percent_positives_lifts[cutoff]["percent_positive"], 2.5), color='dimgrey', linestyle='--', linewidth=1.25)
                        ax_hist.axvline(x=np.percentile(bootstrapped_percent_positives_lifts[cutoff]["percent_positive"], 97.5), color='dimgrey', linestyle='--', linewidth=1.25)
                        legend_patch = Patch(color=cutoffs_to_colors[cutoff], label=f'{model_name} (Percent Positives = {np.mean(bootstrapped_percent_positives_lifts[cutoff]["percent_positive"]):.3f}, CI: {np.percentile(bootstrapped_percent_positives_lifts[cutoff]["percent_positive"], 2.5):.3f}-{np.percentile(bootstrapped_percent_positives_lifts[cutoff]["percent_positive"], 97.5):.3f})')
                        ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

                        # Save the histogram image
                        histogram_image_name = model_name + f'_percent_positives_{cutoff:.0%}_cutoff_histogram.png'
                        plt.savefig(histogram_image_name, dpi=300)
                        plt.close(fig_hist)

                    # Plotting the histogram for bootstrapped lifts per cutoff
                    for cutoff in cutoffs_to_colors.keys():
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.hist(bootstrapped_percent_positives_lifts[cutoff]['lift'], bins=12, color=cutoffs_to_colors[cutoff], ec="k")
                        ax_hist.set_title(f'Distribution of Lifts Using {self.number_of_bootstrap_samples} Bootstraps at {cutoff:.0%} Cutoff')
                        ax_hist.set_xlabel('Lift')
                        ax_hist.set_ylabel('Frequency')
                        fig_hist.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
                        ax_hist.axvline(x=np.percentile(bootstrapped_percent_positives_lifts[cutoff]["lift"], 2.5), color='dimgrey', linestyle='--', linewidth=1.25)
                        ax_hist.axvline(x=np.percentile(bootstrapped_percent_positives_lifts[cutoff]["lift"], 97.5), color='dimgrey', linestyle='--', linewidth=1.25)
                        legend_patch = Patch(color=cutoffs_to_colors[cutoff], label=f'{model_name} (Lift = {np.mean(bootstrapped_percent_positives_lifts[cutoff]["lift"]):.3f}, CI: {np.percentile(bootstrapped_percent_positives_lifts[cutoff]["lift"], 2.5):.3f}-{np.percentile(bootstrapped_percent_positives_lifts[cutoff]["lift"], 97.5):.3f})')
                        ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

                        # Save the histogram image
                        histogram_image_name = model_name + f'_lift_{cutoff:.0%}_cutoff_histogram.png'
                        plt.savefig(histogram_image_name, dpi=300)
                        plt.close(fig_hist)

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
                common_thresholds = np.linspace(0, 1, 100)
                bootstrapped_precisions = []

                for i in range(self.number_of_bootstrap_samples):
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

    @staticmethod
    def plot_decile_patient_prediction_uncertainty(lowest=False, middle=False, highest=False):
        list_of_results_file_paths = []
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_first_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_second_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_third_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_fourth_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_fifth_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_sixth_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_seventh_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_eighth_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_ninth_seed_results.json'))
        list_of_results_file_paths.append(os.path.abspath('..\\predictions\\xgb_tenth_seed_results.json'))
        list_of_results_dfs = []
        for results_file_path in list_of_results_file_paths:
            with open(results_file_path, 'r') as file:
                results_data = json.load(file)
                df = pd.DataFrame(next(iter(results_data.values())))
                df['color'] = df['y_test'].map({0: 'blue', 1: 'red'})
                list_of_results_dfs.append(df)

        # Concatenate the dataframes along the rows
        df_combined = pd.concat(list_of_results_dfs)

        # Group by 'patient_id' and calculate the mean and standard deviation of the probabilities
        df_stats = df_combined.groupby('patient_id')['probabilities'].agg(['mean', 'std']).reset_index()

        # Since color is consistent per patient_id, you can directly take the first occurrence
        df_stats['color'] = df_combined.groupby('patient_id')['color'].first().reset_index(drop=True)

        # Convert 'patient_id' to a string to treat it as a categorical variable
        df_stats['patient_id'] = df_stats['patient_id'].astype(str)

        # Sort the dataframe by the mean probability
        df_stats_sorted = df_stats.sort_values(by='mean')

        deciles = {'lowest': lowest, 'middle': middle, 'highest': highest}
        relevant_deciles = {decile: include for decile, include in deciles.items() if include}
        for decile, include in relevant_deciles.items():
            if decile == 'lowest':
                decile_patients = df_stats_sorted.head(10)
            elif decile == 'middle':
                median_index = df_stats_sorted['mean'].searchsorted(df_stats_sorted['mean'].median())
                decile_patients = df_stats_sorted.iloc[median_index - 5: median_index + 5]
            else:
                decile_patients = df_stats_sorted.tail(10)

            fig, ax = plt.subplots(figsize=(15, 8))

            # Scatter plot with error bars representing the standard deviation
            for _, row in decile_patients.iterrows():
                ax.errorbar(row['patient_id'], row['mean'], yerr=row['std'], fmt='o', color=row['color'], capsize=5)

            ax.set_title(f'{decile.title()} Decile Patient Prediction Uncertainty')
            ax.set_xlabel('Patient ID')
            ax.set_ylabel('Probability (± Standard Deviation)')
            ax.tick_params(axis='x')
            max_difference = 0.03
            if decile == 'lowest' or decile == 'middle':
                decile_patients_min_value = (decile_patients['mean'] - decile_patients['std']).min()
                ax.set_ylim([decile_patients_min_value - 0.00085, decile_patients_min_value + max_difference])
            elif decile == 'highest':
                decile_patients_max_value = (decile_patients['mean'] + decile_patients['std']).max()
                ax.set_ylim([decile_patients_max_value - max_difference, decile_patients_max_value + 0.00085])

            # Create a custom legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='No Death', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Death', markerfacecolor='red', markersize=10)]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1))

            # Customizing the plot to match the uploaded image
            fig.subplots_adjust(bottom=0.18, left=0.18)
            image_name = f'XGBClassifier_{decile}_decile_patient_prediction_uncertainty.png'
            plt.savefig(image_name, dpi=300)
            plt.close(fig)

    def plot_brier_scores_histogram(self):
        for model_name, df in self.model_results.items():
            y_test = df['y_test']
            probabilities = df['probabilities']

            # Bootstrap
            bootstrapped_brier_scores = []

            for i in range(self.number_of_bootstrap_samples):
                # Sample with replacement
                indices = np.random.choice(np.arange(len(y_test)), size=len(y_test), replace=True)
                y_test_boot = y_test.iloc[indices]
                probabilities_boot = probabilities.iloc[indices]

                brier_score = brier_score_loss(y_test_boot, probabilities_boot)
                bootstrapped_brier_scores.append(brier_score)

            # Calculate confidence interval for Brier Score
            lower_bound_brier_score = np.percentile(bootstrapped_brier_scores, 2.5)
            upper_bound_brier_score = np.percentile(bootstrapped_brier_scores, 97.5)

            for cutoff in self.cutoffs_to_colors.keys():
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(bootstrapped_brier_scores, bins=12, color='lightsteelblue', ec="k")
                ax_hist.set_title(f'Distribution of Brier Scores Using {self.number_of_bootstrap_samples} Bootstraps')
                ax_hist.set_xlabel('Brier Scores')
                ax_hist.set_ylabel('Frequency')
                fig_hist.subplots_adjust(bottom=0.13 + (0.05 * len(self.model_results)), left=0.13 + (0.05 * len(self.model_results)))
                ax_hist.axvline(x=lower_bound_brier_score, color='dimgrey', linestyle='--', linewidth=1.25)
                ax_hist.axvline(x=upper_bound_brier_score, color='dimgrey', linestyle='--', linewidth=1.25)
                legend_patch = Patch(color='lightsteelblue', label=f'{model_name} (Brier Score = {np.mean(bootstrapped_brier_scores):.3f}, CI: {lower_bound_brier_score:.3f}-{upper_bound_brier_score:.3f})')
                ax_hist.legend(handles=[legend_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15))

                # Save the histogram image
                histogram_image_name = model_name + f'_brier_scores_histogram.png'
                plt.savefig(histogram_image_name, dpi=300)
                plt.close(fig_hist)

    def generate_predictions_files(self):
        for model_name, df in self.model_results.items():
            df.drop(columns=['binary_predictions'], inplace=True)
            df = df[['probabilities', 'y_test']]
            df.rename(columns={'y_test': 'labels', 'probabilities': 'predictions'}, inplace=True)
            predictions_file_path = os.path.abspath(f'.\\calibration\\{model_name}_predictions.csv')
            df.to_csv(predictions_file_path, index=False)

    def plot_shap_waterfall(shap_values_path, instance_index):
        """
        Generates a SHAP waterfall plot for a specific instance.
        """
        with open(shap_values_path, 'rb') as f:
            loaded_results = pickle.load(f)

        # Access SHAP values and explainer for a specific fold
        shap_values_fold_1 = loaded_results['fold_1']['shap_values']
        explainer_fold_1 = loaded_results['fold_1']['explainer']

        # # Load the SHAP values
        # with open(shap_values_path, 'rb') as f:
        #     shap_values = pickle.load(f)
        
        # # Load the explainer
        # with open(explainer_path, 'rb') as f:
        #     explainer = pickle.load(f)
        
        # Generate the waterfall plot for the specified instance
        shap.plots.waterfall(shap_values_fold_1[instance_index], max_display=10)

        # Optionally, save the plot to a file or display it
        # For displaying, you might want to remove `show=False` from the plot function
        # plt.savefig('waterfall_plot.png')  # Save the plot as a PNG file
        # plt.show()  # Display the plot in the notebook or a Python script
