import sys
import os
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function returns a ColumnTransformer that applies different transformations to different columns.
        """
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ["gender",
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"]
            
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean= False))
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Creating ColumnTransformer for data transformation")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )


            logging.info("ColumnTransformer created successfully")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function initiates the data transformation process.
        It reads the train and test data, applies transformations, and saves the preprocessor.
        """ 
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read successfully")

            logging.info("Obtaining data transformer object")
            preprocessor = self.get_data_transformer_object()

            target_column = 'math_score'
            drop_columns = [target_column]
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]   

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying transformations to train and test data")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Transformations applied successfully")

            train_arr= np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr= np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]
            logging.info("Train and test arrays created successfully")

            logging.info("Saving preprocessor object")

            save_object(
                self.transformation_config.preprocessor_path,
                obj=preprocessor)
            
            logging.info("Preprocessor object saved successfully")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

