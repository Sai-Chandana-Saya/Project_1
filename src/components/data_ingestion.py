import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.utils import save_object
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder")

            # Splitting the dataset into train and test sets
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test data saved to artifacts folder")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    logging.info("Data Ingestion completed")

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    logging.info("Data Transformation completed")

    modeltrainer = ModelTrainer()
    best_model,model_report = modeltrainer.initiate_model_trainer(train_arr, test_arr)
    logging.info("Model Training completed")


    # Save the model report
    with open('artifacts/model_report.txt', 'w') as f:
        # Write the complete model report
        f.write("Model Evaluation Report:\n")
        f.write("="*50 + "\n")
        for model_name, metrics in model_report.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"R2 Score: {metrics['r2_score']}\n")
            f.write(f"MAE: {metrics['mean_absolute_error']}\n")
            f.write(f"MSE: {metrics['mean_squared_error']}\n")
            f.write("="*50 + "\n\n") 




