# Disease Prediction API (LSTM + Federated Learning)

A FastAPI-based web service that predicts diseases from symptoms using a trained LSTM model with federated learning.

## Endpoint
- POST `/predict` → { "symptoms": ["fever", "headache", "fatigue"] }

## Deployed on Render
Live URL: https://your-app.onrender.com

## Local Testing
```bash
uvicorn app:app --reload
