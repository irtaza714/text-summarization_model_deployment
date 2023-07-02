import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from src.exception import CustomException
from src.logger import logging
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader


@dataclass
class DataTransformationConfig:
    trainos_data_path: str=os.path.join('artifacts',"train_os.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self,train_path,test_path,val_path):

        try:
            train = pd.read_csv(train_path)

            logging.info("Read train data")
            
            test = pd.read_csv(test_path)

            logging.info("Read test data")
            
            val = pd.read_csv(val_path)

            logging.info("Read validation data")

            train = Dataset.from_pandas(train)

            logging.info("train arrow dataset formed")
            
            test = Dataset.from_pandas(test)

            logging.info("test arrow dataset formed")
            
            val = Dataset.from_pandas(val)

            logging.info("validation arrow dataset formed")

            dataset = DatasetDict()

            logging.info("arrow dataset dictionary initiated")
            
            dataset['train'] = train

            logging.info("train split of the dataset dictionary made with the train arrow dataset")
            
            dataset['test'] = test

            logging.info("test split of the dataset dictionary made with the test arrow dataset")
            
            dataset['val'] = val

            logging.info("validation split of the dataset dictionary made with the val arrow dataset")

            logging.info("dataset printed")

            dataset = dataset.rename_columns({ "summary": "title", "document": "text"})

            logging.info("dataset columns renamed")

            dataset = dataset.filter(lambda x: len(x["title"].split()) > 2)

            logging.info("filtering the dataset")

            model_checkpoint = "facebook/bart-large-cnn"

            logging.info("model checkpoint initiated")
            
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            logging.info("tokenizer initiated")

            max_input_length = 1024
            
            max_target_length = 512
            
            def preprocess_function(examples):
                 model_inputs = tokenizer(
                      examples["text"],
                      max_length=max_input_length, truncation=True,
                 )
                 labels = tokenizer(
                      examples["title"], max_length=max_target_length, truncation=True
                 )
                 
                 model_inputs["labels"] = labels["input_ids"]
                 
                 return model_inputs
            
            logging.info("Preprocessing function made for tokenizing train dataset")

            tokenized_datasets = dataset.map(preprocess_function, batched=True)

            logging.info("Tokenized complete dataset")

            model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

            logging.info ("Model initialized")

            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            logging.info ("Data collator initialized")

            tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)

            logging.info ("Columns removed from train split") 

            features = [tokenized_datasets["train"][i] for i in range(2)]

            tokenized_datasets.set_format("torch")

            logging.info ("Dataset format set to torch")

            batch_size = 1

            train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator,
                                          batch_size=batch_size
            )
            logging.info("Train Dataloader initialized")
            
            eval_dataloader = DataLoader(tokenized_datasets["val"], collate_fn=data_collator, batch_size= 1)

            logging.info("Validation Dataloader initialized")

            test_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size= 1)

            logging.info("Test Dataloader initialized")
            
            return (
            train_dataloader,
            eval_dataloader,
            test_dataloader)
        
        except Exception as e:
            raise CustomException(e, sys)