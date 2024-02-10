import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_combined_violin_plots(extracted_data, output_dir, base_filename):
    """
    Creates and saves combined violin plots for the probabilities of each classifier,
    showing distributions for y_test=0 and y_test=1 on the same graph.
    
    Parameters:
    - extracted_data: dict, a dictionary where each key is a classifier name and the value
      is another dictionary with keys '0' and '1', each containing lists of probabilities.
    - output_dir: str, the directory where plots will be saved.
    - base_filename: str, the base name of the source JSON file to include in the plot filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for classifier, data in extracted_data.items():
        # Prepare the DataFrame
        df_list = []
        for y_test, probabilities in data.items():
            temp_df = pd.DataFrame(probabilities, columns=['Probabilities'])
            temp_df['y_test'] = 'Non-Death' if y_test == '0' else 'Hospital Death'
            df_list.append(temp_df)
        
        if df_list:  # If there's data to plot
            combined_df = pd.concat(df_list)
            
            plt.figure(figsize=(8, 6))
            sns.violinplot(x='y_test', y='Probabilities', data=combined_df, split=True,width=1.1,color='#8dd3c7')
            plt.title(f'{classifier}: Distribution of Probabilities by Outcome', size=18)
            plt.ylabel('Probability', size=16)
            plt.xlabel('Outcome', size=16)
            plt.xticks(size=14)
            plt.yticks(size=14)
            plt.ylim(bottom=0,top=1)
            plot_filename = os.path.join(output_dir, f'{base_filename}_{classifier}_combined.png')
            plt.savefig(plot_filename)
            plt.close()

def extract_probabilities_from_json(json_file_path):
    """
    Extracts probabilities from a JSON file with multiple classifiers, separated by y_test values.
    
    Parameters:
    - json_file_path: str, the path to the JSON file.
    
    Returns:
    - A dictionary where each key is a classifier name and the value is another dictionary
      with keys '0' and '1', each containing lists of probabilities for y_test=0 and y_test=1, respectively.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    separated_data = {}
    for key in data.keys():
        classifier_name = key.split('_')[0].strip()
        # Initialize nested dictionary for each classifier
        if classifier_name not in separated_data:
            separated_data[classifier_name] = {'0': [], '1': []}
        
        for item in data[key]:
            y_test = str(item["y_test"])  # Convert y_test to string to use as key
            separated_data[classifier_name][y_test].append(item["probabilities"])
    
    return separated_data

##editor fold

# Directory paths
predictions_dir = 'predictions'
output_dir = 'predictions/output/violin_plot_probabilities/'

# Process all JSON files in the predictions directory
for filename in os.listdir(predictions_dir):
    if filename.endswith('.json'):
        json_file_path = os.path.join(predictions_dir, filename)
        result = extract_probabilities_from_json(json_file_path)
        # Use the base filename (without extension) as part of the plot filename
        base_filename = os.path.splitext(filename)[0]
        create_combined_violin_plots(result, output_dir, base_filename)