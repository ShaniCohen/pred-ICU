import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

height_wegiht_dict={"man":{
    1:[76,10.2],2:[85,12.3],3:[95,14.6],4:[102,16.7],5:[109,18.7],6:[116.1,20.7],7:[121.7,22.9], 8:[127,25.3],9:[132.2,28.1],10:[137.5,31.4],11:[140,32.2],12:[147,37],15:[162,52],
    18:[178,76.1],30:[177,83.3],45:[176,84.5],65:[174,79]

},
'F':{
    1:[75,9.5],2:[84.5,11.8],3:[93.9,14.1],4:[101.6,16],5:[108.4,17.7],6:[114.6,19.5],7:[120.6,21.8], 8:[126.4,24.8],9:[132.2,28.5],10:[138.3,32.5],11:[142,33.7],12:[148,38.7],15:[156,51.8]
    ,18:[165,62.8],30:[164,67.1],45:[162,68.7],65:[162,66]
}}
# reference "researchgeocolorado.org",https://www.researchgate.net/figure/Self-reported-height-weight-and-bMI-by-gender-age-and-year-1998-2002-and-2007_tbl3_223995514



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

def create_separate_histograms(extracted_data, output_dir, base_filename):
    """
    Creates and saves separate histograms for the probabilities of class 0 and class 1 for each classifier,
    displayed as two subfigures within one figure, with different y-axis scales.
    
    Parameters:
    - extracted_data: dict, a dictionary where each key is a classifier name and the value
      is another dictionary with keys '0' and '1', each containing lists of probabilities.
    - output_dir: str, the directory where plots will be saved.
    - base_filename: str, the base name of the source JSON file to include in the plot filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for classifier, data in extracted_data.items():
        # Create a figure with two subplots (side by side) without sharing the y-axis
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{classifier}: Distributions for Outcomes', size=20)
        
        # Plot histogram for class 0
        sns.histplot(data=pd.DataFrame(data['0'], columns=['Probabilities']), ax=axes[0], color="skyblue", kde=False,legend=False,bins=20)
        axes[0].set_title('Non-Death', size=18)
        axes[0].set_xlabel('Probability', size=16)
        axes[0].set_ylabel('Frequency', size=16)
        axes[0].tick_params(axis='both', which='major', labelsize=14)
        
        # Plot histogram for class 1
        sns.histplot(data=pd.DataFrame(data['1'], columns=['Probabilities']), ax=axes[1], color="skyblue", kde=False,legend=False,bins=20)
        axes[1].set_title('Hospital Death', size=18)
        axes[1].set_xlabel('Probability', size=16)
        axes[1].set_ylabel('Frequency', size=16)
        axes[1].tick_params(axis='both', which='major', labelsize=14)

        # Save the plot
        plot_filename = os.path.join(output_dir, f'{base_filename}_{classifier}_separate_histograms.png')
        plt.savefig(plot_filename)
        plt.close()

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
        create_separate_histograms(result, output_dir, base_filename)