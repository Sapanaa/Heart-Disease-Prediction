import pandas as pd
import sys
from src.utils.load_object import load_object
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.model_path = "artifacts/model.pkl"

    def predict(self, input_df):
        
        """
        input_data: pandas DataFrame with raw features
        returns: model prediction
        """
        try:
            logging.info("Predict pipeline started")
            # Load preprocessor and model
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            logging.info("Predict pipeline model added")
            # Transform input data
            data_transformed = preprocessor.transform(input_df)
            logging.info("Predict pipeline transformed data added")
            # Make prediction
            prediction = model.predict(data_transformed)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, input_df):
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            data_transformed = preprocessor.transform(input_df)
            prediction = model.predict_proba(data_transformed)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)