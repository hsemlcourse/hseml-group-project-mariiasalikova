import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Ethereum Fraud Detection API", description="API для скоринга Ethereum-адресов")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None
    
class PredictRequest(BaseModel):
    features: dict

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Ethereum Fraud Detection API is running"}

@app.post("/predict")
def predict_fraud(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена на сервере.")

    try:
        # Преобразуем входящий JSON в DataFrame (одна строка)
        df = pd.DataFrame([request.features])

        # Получаем предсказания
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "is_fraud": bool(prediction),
            "fraud_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    