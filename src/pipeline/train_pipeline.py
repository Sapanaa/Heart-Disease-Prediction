import sys
from src.components.data_ingestion import DataIngestion
from src.logger import logger
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.dat_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
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

            # Step 2: Data Transformation
            validation = DataValidation()
            validation.validate_all_columns(train_path, test_path)


            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
            transformation.initiate_data_transformation(
                train_path, test_path
            )
            )
            logger.info(f" Preprocessor: {preprocessor_path}")


            # Step 3: Model Training
            logger.info("Model training started")
            trainer = ModelTrainer()
            model_path = trainer.initiate_model_trainer(train_arr, test_arr)

            logger.info(f"Model training completed. Model: {model_path}")

            evaluator = ModelEvaluation()
            is_accepted, metrics = evaluator.evaluate_model(
                model_path,test_arr)

            if not is_accepted:
                raise Exception("Model did not meet acceptance criteria")

            logger.info(f"Model evaluation completed. Metrics: {metrics}")
        except Exception as e:
            logger.error("Error occurred in training pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
