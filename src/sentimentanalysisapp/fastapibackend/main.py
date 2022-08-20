import uvicorn
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer,Trainer
import numpy as np

app = FastAPI()

# Load trained model
model_path = "./sentimentanalysismodel"
model_loaded = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
test_trainer = Trainer(model_loaded)

@app.get("/")
def home():
    return {"Hello": "World"}

@app.get("/predict/{text}")
async def prediction(text : str):
    test_pred = test_trainer.predict([tokenizer(text,truncation=True,padding=True)]).predictions
    test_label= np.argmax(test_pred, axis=1)
    print(test_label)
    return {"result":str(test_label[0])}
