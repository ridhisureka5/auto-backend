from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="AutoMind Complaint Classifier API")

# -----------------------------
# CORS Configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Model & Assets
# -----------------------------
try:
    model = tf.keras.models.load_model("complaint_model.h5")

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
# Health Check Route
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
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([data.text])
        padded = pad_sequences(sequence, maxlen=10, padding="post")

        # Predict
        prediction = model.predict(padded)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "complaint": data.text,
            "predicted_issue": index_label.get(predicted_class, "Unknown"),
            "confidence": round(confidence, 4),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))