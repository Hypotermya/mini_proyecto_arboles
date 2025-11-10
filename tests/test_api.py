# tests/test_api.py
from fastapi.testclient import TestClient
from app.api import app  # Importa tu instancia de FastAPI

client = TestClient(app)

def test_predict_endpoint():
    """Prueba el endpoint /predict con un ejemplo de cliente"""
    data = {
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
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200, "El endpoint /predict no respondió correctamente"

    result = response.json()
    assert "prediction" in result, "La respuesta no contiene 'prediction'"
    assert "probability" in result, "La respuesta no contiene 'probability'"
    assert isinstance(result["probability"], float), "La probabilidad debe ser un número flotante"
    assert 0 <= result["probability"] <= 1, "La probabilidad no está entre 0 y 1"
