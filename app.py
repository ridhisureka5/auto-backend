from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("complaint_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_index.pkl", "rb") as f:
    label_index = pickle.load(f)

index_label = {v: k for k, v in label_index.items()}

class Complaint(BaseModel):
    text: str

@app.post("/classify")
def classify_complaint(data: Complaint):

    sequence = tokenizer.texts_to_sequences([data.text])
    padded = pad_sequences(sequence, maxlen=10, padding='post')

    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return {
        "complaint": data.text,
        "predicted_issue": index_label[predicted_class],
        "confidence": confidence
    }