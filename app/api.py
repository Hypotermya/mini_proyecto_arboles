from fastapi import FastAPI
import joblib
import pandas as pd
from app.schemas import CustomerData

app = FastAPI(title="Telco Churn Predictor", version="1.0")

# Cargar modelo entrenado
model = joblib.load("app/model.joblib")


@app.get("/")
def home():
    msg = "API de predicción de churn de clientes de Telco está activa"
    return {"message": msg}


@app.post("/predict")
def predict(data: CustomerData):
    # Convertir input en DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Realizar predicción
    pred = model.predict(input_df)[0]
    # Probabilidad de churn (clase 1)
    proba = model.predict_proba(input_df)[0][1]

    # Interpretar la clase
    pred_label = "Churn" if pred == 1 else "No Churn"

    return {
        "prediction": pred_label,
        "probability": round(float(proba), 3),
    }
