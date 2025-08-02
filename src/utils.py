import os
import sys

import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves the object to the specified file path using pandas.
    """ 
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pd.to_pickle(obj, file)
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)