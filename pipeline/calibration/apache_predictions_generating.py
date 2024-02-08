import os
import pandas as pd


def main():
    # Define the path for the output file
    output_file_path = os.path.abspath('apache_predictions.csv')

    # Check if the output file already exists
    if not os.path.exists(output_file_path):
        # Use an absolute path to locate the training data file
        training_data_file_path = os.path.abspath('..\\..\\data\\training_v2.csv')
        training_df = pd.read_csv(training_data_file_path)

        # Select relevant columns
        apache_results_df = training_df[['apache_4a_hospital_death_prob', 'hospital_death']]

        # Filter out unwanted rows and explicitly create a copy to avoid SettingWithCopyWarning
        cleaned_apache_results_df = apache_results_df[(apache_results_df['apache_4a_hospital_death_prob'] != -1) & (apache_results_df['apache_4a_hospital_death_prob'].notna())].copy()

        # Rename columns without risking a SettingWithCopyWarning
        cleaned_apache_results_df.rename(columns={'apache_4a_hospital_death_prob': 'preds', 'hospital_death': 'labels'}, inplace=True)

        # Save the cleaned DataFrame to a CSV file
        cleaned_apache_results_df.to_csv(output_file_path, index=False)
    else:
        print("Output file already exists.")


if __name__ == '__main__':
    main()
