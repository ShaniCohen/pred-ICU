import json
import numpy as np
import glob
from statistics import mode

# Assuming you're running this script in the project folder
folder_path = './'  # Current directory
json_file_paths = glob.glob(f'{folder_path}/*.json')

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

all_predictions = []
for file_path in json_file_paths:
    data = load_json_file(file_path)
    for key in data:
        all_predictions.extend(data[key])

# Initialize dictionaries to store various attributes by patient
probabilities_by_patient = {}
y_test_by_patient = {}
binary_predictions_by_patient = {}

for prediction in all_predictions:
    patient_id = prediction["patient_id"]
    probabilities_by_patient.setdefault(patient_id, []).append(prediction["probabilities"])
    y_test_by_patient.setdefault(patient_id, []).append(prediction["y_test"])
    binary_predictions_by_patient.setdefault(patient_id, []).append(prediction["binary_predictions"])

# Calculate Median Probabilities and aggregate y_test, binary_predictions
output_data = []
for patient_id in probabilities_by_patient:
    median_probability = np.median(probabilities_by_patient[patient_id])
    try:
        # Assuming y_test and binary_predictions are consistent, take the first entry
        y_test = mode(y_test_by_patient[patient_id])
        binary_prediction = mode(binary_predictions_by_patient[patient_id])
    except:
        print(f"Inconsistent values found for patient {patient_id}. Please check the data.")
        continue  # or handle the inconsistency as needed
    
    patient_data = {
        "y_test": y_test,
        "binary_predictions": binary_prediction,
        "probabilities": median_probability,
        "patient_id": patient_id,
    }
    output_data.append(patient_data)

# Construct the final structure with an example configuration as the key
final_output = {
    "XGBClassifier_{'objective': 'binary:logistic', ... 'verbose_eval': False}": output_data
}

# Store the results in a JSON file
output_file_path = f'{folder_path}/final_output.json'
with open(output_file_path, 'w') as output_file:
    json.dump(final_output, output_file, indent=4)

print(f"Output JSON file created at: {output_file_path}")
