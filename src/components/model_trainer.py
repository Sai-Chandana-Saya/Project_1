import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Training models")

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0)
            }

            model_report = {}
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                model_report[model_name] = {
                    'r2_score': r2,
                    'mean_absolute_error': mae,
                    'mean_squared_error': mse,
                    'root_mean_squared_error': rmse
                }
                logging.info(f"{model_name} - R2: {r2}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}")  

            best_model_name = max(model_report, key=lambda x: model_report[x]['r2_score'])
            best_model = models[best_model_name]
            logging.info(f"Best model: {best_model_name} with R2 score: {model_report[best_model_name]['r2_score']}")   

            save_object(self.model_trainer_config.model_path, best_model)
            logging.info(f"Model saved at {self.model_trainer_config.model_path}")

            return best_model, model_report
        
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(e, sys)   
        

