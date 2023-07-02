import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    val_data_path: str=os.path.join('artifacts',"val.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")
        
        try:

            df = pd.read_csv('notebook/multi_news_4000_clean.csv')

            logging.info('Read the dataset as dataframe')

            df = df.dropna()

            logging.info('Dropping Empty Rows')

            df['document'] = df['document'].str.replace('\n', '')

            logging.info('Replacing newline with empty string in document column') 
            
            df['summary'] = df['summary'].str.replace('\n', '')

            logging.info('Replacing newline with empty string in summary column')

            df = df.drop_duplicates()

            logging.info('Dropping duplicates from dataframe') 

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            logging.info("directory made for train dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)

            logging.info("directory made for test dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.val_data_path),exist_ok=True)

            logging.info("directory made for val dataset")

            train_old, test = train_test_split(df, test_size = 0.2, random_state = 1)

            logging.info('Splitting the dataset into train and test')

            train, val =  train_test_split(train_old, test_size = 0.2, random_state = 1)
            
            logging.info("Validation Splits Also Made")

            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("Train Set Saved")

            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Test Set Saved")
            
            val.to_csv(self.ingestion_config.val_data_path,index=False,header=True)

            logging.info("Validation Set Saved")

            logging.info("Ingestion of the data has completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.val_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    # obj.initiate_data_ingestion()

    train_data, test_data, val_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_dataloader, eval_dataloader, test_dataloader = data_transformation.initiate_data_transformation(train_data, test_data, val_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_dataloader, eval_dataloader))