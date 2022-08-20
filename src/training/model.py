import pandas as pd 
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import *
import datasets 
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer,\
get_scheduler,DataCollatorWithPadding
import evaluate
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score,classification_report


# Data loading 
tweet_df = pd.read_csv("../scraping/labelled_COVID-19_vaccine.csv",index_col=None, header=0, engine='python',usecols=['text','labels'])

# make the dataset balanced before splitting
tweet_df_2 = tweet_df[tweet_df['labels']==2]
tweet_df_1 = tweet_df[tweet_df['labels'] == 1]
tweet_df_0 = tweet_df[tweet_df['labels'] == 0]
min_len = min(len(tweet_df_2),len(tweet_df_1),len(tweet_df_0))
tweet_df = pd.concat([tweet_df_0.iloc[:min_len],tweet_df_1.iloc[:min_len],tweet_df_2.iloc[:min_len]])
tweet_df = shuffle(tweet_df,random_state=42)

# Data spllitting 
X = list(tweet_df["text"])
y = list(tweet_df["labels"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
X_train_tokenized = tokenizer(X_train,truncation=True,max_length=512)
X_val_tokenized = tokenizer(X_val,truncation=True,max_length=512)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = TweetDataset(X_train_tokenized, y_train)
val_dataset = TweetDataset(X_val_tokenized, y_val)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds,average='macro')
    precision = precision_score(y_true=labels, y_pred=preds,average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds,average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

device = "cuda" if torch.cuda.is_available() else "cpu"

# model and tokenizer 
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=3).to(device).to(device)

# Trainer
args = TrainingArguments(
    output_dir="./test_trainer",
    learning_rate=2e-5,
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics)

# Fine tuning model
trainer.train()

