import os
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
)
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from typing import Callable, Dict
import sys 

sys.path.append(os.path.abspath('../..'))
import config.model_config as model_config

def compute_metrics(p: EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
  preds = np.argmax(preds, axis=1)
  #result = metric.compute(predictions=preds, references=p.label_ids)
  return {"acc": (preds == p.label_ids).astype(np.float32).mean().item()}


# Create dataset
class TweetDataset(Dataset):
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
        

def tune_transformer(num_samples=8, gpus_per_trial=1, cpus_per_trial=2,
           num_labels=3,
           train_file = '/train.csv',test_file= '/test.csv',
           data_dir_name = model_config.DATA_DIR,log_dir_name = model_config.LOGS_path,
           ray_results_dir=model_config.RAY_path,model_dir_name = model_config.MODELS_path):
     
    if not os.path.exists(data_dir_name):
        os.mkdir(data_dir_name, 0o755)

    train_file = data_dir_name+train_file
    test_file = data_dir_name+test_file

    # Change these as needed.
    #task_name = "sentiment" 

    model_name = "distilbert-base-uncased"

    # Download and cache tokenizer, model, and features
    print("Downloading and caching Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,num_labels=num_labels).to(device)

    def load_data():
      train_df = pd.read_csv(train_file,index_col=None, header=0,engine='python')
      val_df = pd.read_csv(test_file,index_col=None, header=0,engine='python')
      X_train,y_train = train_df[text_col].tolist(),train_df[labels_col].tolist()
      X_val,y_val = val_df[text_col].tolist(),val_df[labels_col].tolist()

      # Tokenization
      X_train_tokenized = tokenizer(X_train,truncation=True,padding=True)
      X_val_tokenized = tokenizer(X_val,truncation=True,padding=True)

      train_dataset = TweetDataset(X_train_tokenized, y_train)
      val_dataset = TweetDataset(X_val_tokenized, y_val)

      return train_dataset,val_dataset


    # Download data.
    train_dataset,eval_dataset = load_data()

    training_args = TrainingArguments(
        output_dir=model_dir_name,
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,
        weight_decay=0.1,  # config
        logging_dir=log_dir_name,
        skip_memory_metrics=True,
        report_to="none",
    )


    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    tune_config = {
        "output_dir":model_dir_name,
        "per_device_train_batch_size": tune.choice([16, 32]),
        "num_train_epochs": tune.choice([2, 3, 4]),
        "weight_decay": tune.uniform(0.0, 0.3),
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "max_steps": -1
    }


    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_acc",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            "per_device_train_batch_size": [16, 32],
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
    )

    
    best_trial = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=num_samples,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop= None,
        progress_reporter=reporter,
        local_dir=ray_results_dir,
        log_to_file=True,
    )


if __name__ == "__main__":
  text_col,labels_col = "text","labels"
  device = "cuda" if torch.cuda.is_available() else "cpu" 

  ray.init(ignore_reinit_error=True)
  tune_transformer(num_samples=8, gpus_per_trial=0,cpus_per_trial=2)
