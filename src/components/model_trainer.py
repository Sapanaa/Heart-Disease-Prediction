import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model  import LogisticRegression
from src.utils import save_object
import dill
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Starting model training and validation")

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                    "LogisticRegression": LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced"
                    ),
                    "RandomForest": RandomForestClassifier(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42,
                        class_weight="balanced"
                    ),
                    "SVM": SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced"
                    ),
                    "Naive Bayes": GaussianNB(),
                    "Decision Tree": DecisionTreeClassifier(
                        random_state=42,
                        class_weight="balanced"
                    ),
                    "KNN": KNeighborsClassifier()

                }

            best_model = None
            best_cv_score = -1

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                cv_scores = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=5,
                        scoring="f1"
                    )

                mean_cv = cv_scores.mean()
                std_cv = cv_scores.std()

                logging.info(
                        f"{model_name} CV F1-score: {mean_cv:.4f} ± {std_cv:.4f}"
                    )

                if mean_cv > best_cv_score:
                    best_cv_score = mean_cv
                    best_model = model

                # Train best model on full training data
            best_model.fit(X_train, y_train)

                # Final evaluation on test set (used ONCE)
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            test_f1 = f1_score(y_test, y_pred)
            test_roc = roc_auc_score(y_test, y_proba)

            logging.info(f"Final Test F1-score: {test_f1:.4f}")
            logging.info(f"Final Test ROC-AUC: {test_roc:.4f}")

                # Save model
            os.makedirs("artifacts", exist_ok=True)
            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            logging.info("Best model saved successfully")

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
