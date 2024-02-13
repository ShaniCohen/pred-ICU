import pandas as pd

class ModelCalibration:
    def __init__(self, predictions_csv: str, group_columns: list[str]) -> None:
        self.model_results = pd.read_csv(predictions_csv)
        self.groups = []
        for group in group_columns:
            self.groups.extend(self.model_results[group].unique())
    
    def get_calibrations_in_the_large(self):
        pass
