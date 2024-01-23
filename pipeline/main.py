from logging_config import setup_logging
from data_handler import DataHandler
import os
from preprocessing import Preprocessing
from sklearn.preprocessing import MinMaxScaler
from model_handler import ModelHandler
import xgboost as xgb
from ml_classification_pipeline import MLClassificationPipeline


# Main function of the pred-ICU Pipeline
def main():
    # Setup logging
    setup_logging()

    # Create objects
    data_handler = DataHandler(file_path=os.path.abspath('..\\data\\training_v2.csv'))
    preprocessing = Preprocessing(scaler=MinMaxScaler())
    model_handler = ModelHandler(model=xgb.XGBClassifier())

    # Create pipeline object
    pipeline = MLClassificationPipeline(data_handler=data_handler, preprocessing=preprocessing, model_handler=model_handler)

    # Run pipeline
    pipeline.run_pipeline()


if __name__ == '__main__':
    main()
