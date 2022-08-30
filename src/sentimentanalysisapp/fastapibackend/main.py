import uvicorn
from fastapi import FastAPI
import numpy as np
import inference as inf 

app = FastAPI()

@app.get("/")
def home():
    return {"Hello": "World"}

@app.get("/predict/{text}")
async def prediction(text : str):
    label= inf.pipeline(text)
    return {"result":str(label)}