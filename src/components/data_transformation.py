from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
import os
import sys
import numpy as np
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_preprocessor(self):
        try:
            
            numeric_features = [
                "Age", "Cholesterol", "Blood Pressure",
                "Heart Rate", "Exercise Hours",
                "Stress Level", "Blood Sugar"
            ]

            categorical_features = [
                "Gender", "Smoking", "Alcohol Intake",
                "Family History", "Diabetes", "Obesity",
                "Exercise Induced Angina", "Chest Pain Type"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_features),
                    ("cat", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        logging.info('Data Transformation methods Starts')
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "Heart Disease"

            X_train = train_df.drop(columns=[target_column], axis=1)
            X_test = test_df.drop(columns=[target_column], axis=1)

            y_train = train_df[target_column]
            y_test = test_df[target_column]

            logging.info("X_train and X_test initiated")

            preprocessor = self.get_preprocessor()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            logging.info("X_train and X_test transformed")

            train_arr = np.c_[X_train_transformed, y_train.values]
            test_arr = np.c_[X_test_transformed, y_test.values]

            os.makedirs("artifacts", exist_ok=True)
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info("Data transformation completed")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e,sys)
                    