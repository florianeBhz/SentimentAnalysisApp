import pandas as pd 
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datasets import metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer,\
get_scheduler,DataCollatorWithPadding
from sklearn.metrics import accuracy_score
import sys 
import os 

sys.path.append(os.path.abspath('../..'))
import config.model_config as model_config

text_col,labels_col = model_config.TEXT_COL, model_config.LABELS_COL
labels = list(model_config.LABELS_DICT.keys())
num_labels = len(labels)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Create dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item[labels_col] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Data loading 
def load_data(train_file,test_file):
    train_df = pd.read_csv(model_config.TRAIN_FILE,index_col=None, header=0,engine='python')
    val_df = pd.read_csv(model_config.TEST_FILE,index_col=None, header=0,engine='python')
    X_train,y_train = train_df[text_col].tolist(),train_df[labels_col].tolist()
    X_val,y_val = val_df[text_col].tolist(),val_df[labels_col].tolist()

    # Tokenization
    X_train_tokenized = tokenizer(X_train,truncation=True,max_length=512)
    X_val_tokenized = tokenizer(X_val,truncation=True,max_length=512)

    train_dataset = TweetDataset(X_train_tokenized, y_train)
    val_dataset = TweetDataset(X_val_tokenized, y_val)

    return train_dataset,val_dataset


# model 
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name
    ,num_labels=num_labels,return_dict=True).to(device)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=-1)
    #accuracy = accuracy_score(y_true=labels, y_pred=preds)
    return metric.compute(predictions=preds, references=labels)


device = "cuda" if torch.cuda.is_available() else "cpu" 
task = "eng-tweet-sentiment-analysis"
metric_name = "accuracy"

# Trainer
training_args = TrainingArguments(
    model_config.MODEL_DIR,
    evaluation_strategy = "epoch",   
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

train_dataset, val_dataset = load_data(model_config.TRAIN_FILE,model_config.TEST_FILE)

trainer = Trainer(
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init=model_init)






