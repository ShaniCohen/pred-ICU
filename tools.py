import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_violin_plots_for_classifiers(extracted_data, output_dir, base_filename):
    """
    Creates and saves violin plots for the probabilities of each classifier.
    Each plot will have the classifier's name as the title and will be saved
    with a filename that includes the classifier name and the base name of the source JSON file.
    
    Parameters:
    - extracted_data: dict, a dictionary where each key is a classifier name and the value
      is a list of probabilities.
    - output_dir: str, the directory where plots will be saved.
    - base_filename: str, the base name of the source JSON file to include in the plot filename.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for classifier, probabilities in extracted_data.items():
        # Convert the probabilities list into a DataFrame for visualization
        probabilities_df = pd.DataFrame(probabilities, columns=['Probabilities'])
        
        # Plot the violin plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=probabilities_df, y='Probabilities')
        plt.title(f'Distribution of Probabilities: {classifier}',size=18)
        plt.ylabel('Probability',size=16)
        plt.yticks(size=16)
        
        # Save the plot with a filename that includes the classifier name and source JSON file name
        plot_filename = os.path.join(output_dir, f'{base_filename}_{classifier}.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the figure


def extract_probabilities_from_json(json_file_path):
    """
    Extracts probabilities from a JSON file with multiple classifiers.
    
    Parameters:
    - json_file_path: str, the path to the JSON file.
    
    Returns:
    - A dictionary where each key is a classifier name and the value is a list of probabilities.
    """
    # Load the JSON content
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    probabilities_data = {}
    # Iterate over all keys (classifiers) in the JSON file
    for key in data.keys():
        # Extract the classifier name before the '_'
        classifier_name = key.split('_')[0].strip()
        # Extract probabilities for each item associated with this classifier
        probabilities = [item["probabilities"] for item in data[key]]
        
        # If the classifier name already exists, extend the list of probabilities
        if classifier_name in probabilities_data:
            probabilities_data[classifier_name].extend(probabilities)
        else:
            probabilities_data[classifier_name] = probabilities
    
    return probabilities_data

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
        create_violin_plots_for_classifiers(result, output_dir, base_filename)