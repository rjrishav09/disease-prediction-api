import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(
    title="Disease Prediction API",
    version="1.0",
    description="Predict disease from symptoms using ML model"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load model & encoders (once)
# -----------------------------
try:
    model = load_model("global_model.h5")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

try:
    with open("sym_idx.pkl", "rb") as f:
        sym_idx = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Encoder loading failed: {e}")

NUM_SYMPTOMS = len(sym_idx)
NUM_CLASSES = len(le.classes_)

# -----------------------------
# Request schema
# -----------------------------
class SymptomsInput(BaseModel):
    symptoms: list[str]

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Disease Prediction API is live",
        "endpoints": ["/predict", "/docs"]
    }

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(input: SymptomsInput):
    if not input.symptoms:
        raise HTTPException(
            status_code=400,
            detail="Symptoms list cannot be empty"
        )

    # Normalize input
    symptoms = [s.strip().lower() for s in input.symptoms]

    # Build input vector
    input_vec = np.zeros(NUM_SYMPTOMS, dtype=np.float32)
    valid_symptoms = []

    for sym in symptoms:
        if sym in sym_idx:
            input_vec[sym_idx[sym]] = 1.0
            valid_symptoms.append(sym)

    if not valid_symptoms:
        raise HTTPException(
            status_code=400,
            detail="No valid symptoms provided"
        )

    # Ensure correct input shape (MATCH TRAINING)
    input_vec = input_vec.reshape(1, NUM_SYMPTOMS, 1)

    # Predict
    prediction = model.predict(input_vec, verbose=0)

    if prediction is None or prediction.size == 0:
        raise HTTPException(
            status_code=500,
            detail="Model returned empty prediction"
        )

    disease_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    if disease_idx >= NUM_CLASSES:
        raise HTTPException(
            status_code=500,
            detail="Prediction index out of label range"
        )

    disease = le.classes_[disease_idx]

    return {
        "predicted_disease": disease,
        "confidence": round(confidence, 4),
        "valid_symptoms_used": valid_symptoms,
        "total_symptoms_sent": len(symptoms),
        "message": "Prediction successful"
    }
