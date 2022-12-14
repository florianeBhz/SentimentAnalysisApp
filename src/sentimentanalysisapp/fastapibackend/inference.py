from distutils.command.config import config
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

"""
import boto3
import config

AWS_ACCESS_KEY_ID = config.AWS_ACCESS_KEY_ID 
AWS_SECRET_ACCESS_KEY = config.AWS_SECRET_ACCESS_KEY 
BUCKET_NAME = config.BUCKET_NAME
FILE_NAME = "sentimentanalysismodel/config.json"
DEST = "./config.json"
s3 = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY_ID , aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
""" 

# Load trained model
model_path = "./sentimentanalysismodel"
model_loaded = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def pipeline(text):
    encodings = tokenizer(text,truncation=True,padding=True,return_tensors='pt')
    pred = model_loaded(**encodings)
    pred_np = pred[0][0].detach().numpy()
    res = np.argmax(pred_np)
    return res
