from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import random

app = FastAPI(title="AutoMind Complaint Classifier API")

# -----------------------------
# CORS Configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# PyTorch Anomaly Model
# -----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

anomaly_model = AutoEncoder()
anomaly_model.eval()

# -----------------------------
# Load TensorFlow Complaint Model
# -----------------------------
try:
    tf_model = tf.keras.models.load_model("complaint_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_index.pkl", "rb") as f:
        label_index = pickle.load(f)

    index_label = {v: k for k, v in label_index.items()}

except Exception as e:
    print("Error loading model files:", e)
    raise RuntimeError("Model files failed to load")

# -----------------------------
# Request Schema
# -----------------------------
class Complaint(BaseModel):
    text: str

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "AutoMind Backend Live 🚀"}

# -----------------------------
# Classification Route
# -----------------------------
@app.post("/classify")
def classify_complaint(data: Complaint):
    try:
        sequence = tokenizer.texts_to_sequences([data.text])
        padded = pad_sequences(sequence, maxlen=10, padding="post")

        prediction = tf_model.predict(padded)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "complaint": data.text,
            "predicted_issue": index_label.get(predicted_class, "Unknown"),
            "confidence": round(confidence, 4),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Dashboard Route
# -----------------------------
@app.get("/dashboard")
def get_dashboard():

    sample = torch.randn(1, 10)

    with torch.no_grad():
        output = anomaly_model(sample)
        anomaly = torch.mean((sample - output)**2).item()

    component_score = round(anomaly, 2)
    batch_score = round(random.uniform(0.3, 0.8), 2)
    lifecycle = round(random.uniform(0.5, 0.9), 2)
    safety = 0.9

    confidence = (
        0.4 * component_score +
        0.3 * batch_score +
        0.2 * lifecycle +
        0.1 * safety
    )

    return {
        "fleet": {
            "totalVehicles": 12450,
            "activeBatches": 12,
            "criticalAlerts": 5,
            "fleetHealth": 87
        },
        "components": {
            "turbo": component_score,
            "battery": batch_score,
            "brake": round(random.uniform(0.2, 0.5), 2),
            "steering": round(random.uniform(0.3, 0.7), 2)
        },
        "confidence": round(confidence * 100, 2),
        "degradationTrend": [0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        "batchCluster": [
            {"x": 0.2, "y": 0.3},
            {"x": 0.5, "y": 0.6},
            {"x": 0.8, "y": 0.9}
        ]
    }