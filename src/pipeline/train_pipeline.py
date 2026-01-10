import sys
from src.components.data_ingestion import DataIngestion
from src.logger import logger
from src.exception import CustomException


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logger.info("Training pipeline started")

            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            logger.info(f"Data ingestion completed. Train: {train_path}, Test: {test_path}")

        except Exception as e:
            logger.error("Error occurred in training pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
