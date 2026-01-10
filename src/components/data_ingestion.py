import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#from src.components.data_transformation import DataTransformation
#from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        Method Name: initiate_data_ingestion
        Description: This method is responsible for splitting the data into train and test.
        Output: train_data_path, test_data_path (str)
        On Failure: Raise CustomException
        '''
        logging.info('Data Ingestion methods Starts')
        try:
            df=pd.read_csv('data/raw/heart_disease_dataset.csv') #As heart-disease-prediction is from where it starts
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #saved the raw file so that the original file will not be effected

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42, stratify=df["Heart Disease"])

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of Data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        


