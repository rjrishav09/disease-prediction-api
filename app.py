import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Disease Prediction API", version="1.0")

# Allow Flutter web/mobile to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
model = load_model('global_model.h5')
with open('sym_idx.pkl', 'rb') as f:
    sym_idx = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

class SymptomsInput(BaseModel):
    symptoms: list[str]

@app.get("/")
def home():
    return {"message": "Disease Prediction API is live! Go to /docs for testing."}

@app.post("/predict")
def predict(input: SymptomsInput):
    symptoms = [s.strip().lower() for s in input.symptoms]  # Normalize input
    
    input_vec = np.zeros((1, len(sym_idx)))
    for sym in symptoms:
        if sym in sym_idx:
            input_vec[0, sym_idx[sym]] = 1
        else:
            return {"error": f"Unknown symptom: '{sym}'"}

    input_vec = input_vec.reshape(1, len(sym_idx), 1)

    prediction = model.predict(input_vec, verbose=0)
    disease_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    disease = le.inverse_transform([disease_idx])[0]

    return {
        "predicted_disease": disease,
        "confidence": round(confidence, 4),
        "message": "Prediction successful"
    }
