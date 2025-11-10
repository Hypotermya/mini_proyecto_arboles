# Predicción de Churn en Telecomunicaciones

## Descripción
Proyecto de ML para predecir abandono de clientes (churn) usando Random Forest, XGBoost y técnicas de interpretabilidad.

## Componentes
- `notebooks/`: Análisis, entrenamiento y explicabilidad.
- `app/api.py`: API REST para predicciones.
- `Dockerfile`: Configuración para despliegue.

## Instalación
```bash
pip install -r requirements.txt
uvicorn app.api:app --reload  # Desarrollo local
```

## Docker
```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

## Endpoints
- `POST /predict`: Recibe JSON con features del cliente, retorna probabilidad de churn.