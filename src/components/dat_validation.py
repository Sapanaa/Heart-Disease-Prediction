import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataValidationConfig:
    validation_report_path: str = os.path.join(
        "artifacts", "validation_report.txt"
    )

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_all_columns(self, train_path, test_path):
        try:
            logging.info("Starting data validation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # 1️⃣ Schema consistency
            train_columns = set(train_df.columns)
            test_columns = set(test_df.columns)

            if train_columns != test_columns:
                raise ValueError(
                    f"Schema mismatch. "
                    f"Train-only columns: {train_columns - test_columns}, "
                    f"Test-only columns: {test_columns - train_columns}"
                )

            # 2️⃣ Target validation
            if "Heart Disease" not in train_columns:
                raise ValueError("Target column 'Heart Disease' missing")

            if not set(train_df["Heart Disease"].unique()).issubset({0, 1}):
                raise ValueError("Target column must be binary (0/1)")

            # 3️⃣ Empty dataset check
            if train_df.empty or test_df.empty:
                raise ValueError("Train or test dataset is empty")

            # Save validation success
            os.makedirs("artifacts", exist_ok=True)
            with open(self.validation_config.validation_report_path, "w") as f:
                f.write("Data validation successful\n")

            logging.info("Data validation passed")
            return True

        except Exception as e:
            logging.error("Data validation failed")

            with open(self.validation_config.validation_report_path, "w") as f:
                f.write(f"Validation failed: {str(e)}\n")

            raise CustomException(e, sys)
