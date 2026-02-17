from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import SalesFeatures, SalesPrediction
from model.train import train_and_save_model

MODEL_PATH = Path("artifacts/sales_model.joblib")

app = FastAPI(title="Sales Forecasting API", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    if not MODEL_PATH.exists():
        train_and_save_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=SalesPrediction)
def predict(payload: SalesFeatures) -> SalesPrediction:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifact not found.")

    model = joblib.load(MODEL_PATH)
    features_df = pd.DataFrame([payload.model_dump()])
    prediction = float(model.predict(features_df)[0])

    return SalesPrediction(predicted_weekly_sales=round(prediction, 2))
