import sys
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            trained_model = "trained_model/"
            tokenizer = AutoTokenizer.from_pretrained(trained_model)
            summarizer = pipeline("summarization", model=trained_model, tokenizer=tokenizer)

            review_truncated = features[:1024]
            result = summarizer(review_truncated)
            results = result[0]['summary_text']
            return results

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_input_text(self):
        try:
            return self.text
        except Exception as e:
            raise CustomException(e, sys)