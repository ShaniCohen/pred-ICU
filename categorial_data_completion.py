from typing import Any
from abc import ABC, abstractclassmethod
import pandas as pd
from pandas.api.types import is_string_dtype


class FillMethod(ABC):
    column: str

    def __init__(self, column: str):
        self.column = column
    
    @abstractclassmethod
    def fill(self, data: pd.DataFrame):
        pass


class FillWithMode(FillMethod):

    def __init__(self, column: str):
        super().__init__(column)

    @abstractclassmethod
    def fill(self, data: pd.DataFrame):
        mode_value = data[self.column].mode()[0]
        data[self.column].fillna(mode_value, inplace=True)


class FillWithValue(FillMethod):
    value: Any
    
    def __init__(self, column: str, value: Any):
        super().__init__(column)
        self.value = value
    
    def fill(self, data: pd.DataFrame):
        data[self.column].fillna(self.value, inplace=True)


class FillWithNewCategory(FillMethod):
    def __init__(self, column: str):
        super().__init__(column)

    @abstractclassmethod
    def fill(self, data: pd.DataFrame):
        if (is_string_dtype(data[self.column])):
            data[self.column].fillna('UNKNOWN')
        else:
            data[self.column] = -1


# The categorial columns with missing values and the fill method applied to them
categorial_columns_fill_methods: list[FillMethod] = [
    FillWithMode('gender'), # 25 missing out of 91713
    FillWithValue('ethnicity', 'Other/Unknown'), # 1395 missing out of 91713
    FillWithNewCategory('hospital_admit_source'), # 21409 missing out of 91713
    FillWithNewCategory('icu_admit_source'), # 112 missing out of 91713
    FillWithValue('arf_apache', 0), # 715 missing out of 91713
    # 'gcs_eyes_apache', # 1901 missing out of 91713, this feature relies on APACHE 3 score
    # 'gcs_motor_apache', # 1901 missing out of 91713, this feature relies on APACHE 3 score
    # 'gcs_unable_apache', # 1037 missing out of 91713, this feature relies on APACHE 3 score
    # 'gcs_verbal_apache', # 1901 missing out of 91713, this feature relies on APACHE 3 score
    # 'intubated_apache', # 715 missing out of 91713, this feature relies on APACHE 3 score
    # 'ventilated_apache', # 715 missing out of 91713, this feature relies on APACHE 3 score
    FillWithValue('aids', 0), # 715 missing out of 91713
    FillWithValue('cirrhosis', 0), # 715 missing out of 91713
    FillWithValue('diabetes_mellitus', 0), # 715 missing out of 91713
    FillWithValue('hepatic_failure', 0), # 715 missing out of 91713
    FillWithValue('immunosuppression', 0), # 715 missing out of 91713
    FillWithValue('leukemia', 0), # 715 missing out of 91713
    FillWithValue('lymphoma', 0), # 715 missing out of 91713
    FillWithValue('solid_tumor_with_metastasis', 0), # 715 missing out of 91713
    FillWithNewCategory('apache_3j_bodysystem'), # 1662 missing out of 91713
    FillWithNewCategory('apache_2_bodysystem') # 1662 missing out of 91713
]


def main():
    # Load the dataset
    data_path = '../Data/training_v2.csv.zip'
    data_df = pd.read_csv(data_path)

    for fill_method in categorial_columns_fill_methods:
        fill_method.fill(data_df)
    
    data_df.to_csv('../Data/training_v2_filled_categorial.csv')

