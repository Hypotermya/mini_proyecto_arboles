# tests/test_model.py
import joblib
import numpy as np
import pandas as pd
import pytest

# Carga del modelo entrenado
MODEL_PATH = "app/model.joblib"

@pytest.fixture(scope="module")
def modelo():
    return joblib.load(MODEL_PATH)

def test_modelo_carga_correctamente(modelo):
    """Verifica que el modelo se carga sin errores"""
    assert modelo is not None, "El modelo no se cargó correctamente"

def test_modelo_predice(modelo):
    """Prueba una predicción con datos simulados"""
    data_prueba = pd.DataFrame([{
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.0,
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }])

    pred = modelo.predict(data_prueba)
    prob = modelo.predict_proba(data_prueba)[:, 1]

    assert isinstance(pred[0], (int, np.integer, str)), "La predicción no tiene el formato esperado"
    assert 0 <= prob[0] <= 1, "La probabilidad no está entre 0 y 1"
