import os
import pickle
from src.exception import CustomException
from src.logger import logging
import sys
import dill

def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file:
            obj = dill.load(file)

        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)

    

