import os
import sys
import pickle
import numpy as np
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object

class ModelEvaluationConfig:
    evaluation_report_path: str = os.path.join(
        "artifacts", "model_evaluation.txt"
    )


class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def evaluate_model(self, model_path, test_arr):
        try:
            logging.info("Starting model evaluation")

            # Load model
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }

            # Decision rule
            ACCEPTANCE_THRESHOLD = 0.80
            is_accepted = metrics["roc_auc"] >= ACCEPTANCE_THRESHOLD

            # Save report
            os.makedirs("artifacts", exist_ok=True)
            with open(self.model_evaluation_config.evaluation_report_path, "w") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v:.4f}\n")
                f.write(f"\nModel Accepted: {is_accepted}\n")

            logging.info(f"Model accepted: {is_accepted}")

            return is_accepted, metrics

        except Exception as e:
            raise CustomException(e, sys)
