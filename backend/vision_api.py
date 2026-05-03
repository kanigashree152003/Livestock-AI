from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import json
import requests
import base64
from datetime import datetime

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from ultralytics import YOLO

from sentence_transformers import SentenceTransformer
import faiss
import pickle
# -------------------------------
# INIT
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FLOWISE_URL = "http://localhost:3000/api/v1/prediction/6870a7c7-e164-4d4f-bdd0-6ef37dae71e8"

# -------------------------------
# LOAD RAG
# -------------------------------
print("Loading RAG...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("rag/faiss.index")

with open("rag/documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("✅ RAG loaded")

# -------------------------------
# CNN MODEL
# -------------------------------
def build_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(3, activation="softmax")
    ])
    return model

cnn_model = build_model()
cnn_model.load_weights("model/cnn.weights.h5")

with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

yolo_model = YOLO("model/yolo_model.pt")

# -------------------------------
# RAG FUNCTIONS
# -------------------------------
def retrieve_context(query, k=3):
    q_vec = embed_model.encode([query])
    D, I = index.search(q_vec, k)
    docs = [documents[i] for i in I[0]]
    scores = D[0]
    return docs, scores

def decide_mode(scores):
    if len(scores) == 0:
        return "llm"
    if scores[0] >= 0.75:
        return "rag"
    elif scores[0] >= 0.6:
        return "rag_warning"
    else:
        return "llm"

# -------------------------------
# FLOWISE (🔥 FIXED)
# -------------------------------
def call_flowise(question, mode="general", context=None, disease=None, confidence=None):
    try:

        # 🔥 IMAGE MODE FIRST (IMPORTANT)
        if disease is not None:
            prompt = f"""
You are a livestock veterinary assistant.

Disease: {disease}
Confidence: {confidence}

Explain in a friendly conversational way:
- What this disease is
- Symptoms
- Cause
- Treatment
- Prevention

Do NOT mention other diseases.
Do NOT say you cannot analyze images.
"""

        # 🔥 RAG MODE
        elif mode in ["rag", "rag_warning"]:
            prompt = f"""
You are a livestock veterinary assistant.

Context:
{context}

Question:
{question}

Answer naturally like a vet speaking to a farmer.
Use only one disease. Do not mix diseases.
"""

        # 🔥 GENERAL MODE
        else:
            prompt = f"""
You are a livestock assistant.

Answer clearly:

Question:
{question}
"""

        res = requests.post(FLOWISE_URL, json={"question": prompt})
        data = res.json()

        text = (
            data.get("text")
            or data.get("answer")
            or data.get("response")
            or str(data)
        )

        return {"text": text}

    except Exception as e:
        return {"text": f"Error: {str(e)}"}

# -------------------------------
# PREDICTION
# -------------------------------
def predict_disease(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img, verbose=0)[0]
    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return class_names[class_id], confidence

def draw_yolo_boxes(img):
    results = yolo_model(img)
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    return img

# -------------------------------
# API ROUTES
# -------------------------------
@app.get("/")
def home():
    return {"message": "API running ✅"}

@app.post("/detect_and_explain")
async def detect_and_explain(
    file: UploadFile = File(...),
    query: str = Form(""),
    location: str = Form("unknown")
):

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    disease, confidence = predict_disease(img)

    # YOLO
    img = draw_yolo_boxes(img)

    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode()

    # 🔥 RAG + IMAGE COMBINATION
    combined_query = f"{disease} {query}"

    docs, scores = retrieve_context(combined_query)
    mode = decide_mode(scores)

    context = docs[0] if docs else ""

    agent = call_flowise(
        combined_query,
        mode=mode,
        context=context,
        disease=disease,
        confidence=confidence
    )

    return {
        "disease": disease,
        "confidence": confidence,
        "image": img_base64,
        "agent_response": agent,
        "source": mode,
        "rag_score": float(scores[0]) if len(scores) > 0 else 0,
    }

@app.post("/text_query")
async def text_query(query: str = Form(...)):
    docs, scores = retrieve_context(query)
    mode = decide_mode(scores)

    context = docs[0] if docs else ""

    agent = call_flowise(query, mode=mode, context=context)

    return {
        "agent_response": agent,
        "source": mode,
        "rag_score": float(scores[0]) if len(scores) > 0 else 0
    }