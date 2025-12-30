import os
import time
import json
import base64
import traceback
import tempfile
import threading
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2

# Import kedua loader
from model_utils import load_cnn_model, predict_cnn, load_lstm_model, predict_lstm

# Config
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "temp_videos"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

CNN_IMG_SIZE = 224   # Ukuran buat CNN
LSTM_IMG_SIZE = 128  # Ukuran buat LSTM (Sesuai training)
LSTM_FRAMES = 20     # Frame buat LSTM

# Global State
CNN_MODEL = None
CNN_CLASSES = None
LSTM_MODEL = None
LSTM_CLASSES = None

app = Flask(__name__)
CORS(app)

def ensure_dirs():
    os.makedirs(VIDEOS_DIR, exist_ok=True)

# Helper Base64
def safe_b64_to_bytes(data_b64):
    if "base64," in data_b64:
        data_b64 = data_b64.split("base64,")[1]
    return base64.b64decode(data_b64)

# Load All Models
def load_models():
    global CNN_MODEL, CNN_CLASSES, LSTM_MODEL, LSTM_CLASSES
    print("🔄 Memuat Semua Model...")
    CNN_MODEL, CNN_CLASSES = load_cnn_model()
    LSTM_MODEL, LSTM_CLASSES = load_lstm_model()

# ===========================
# 🛣️ ROUTES
# ===========================

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), 'index.html')

@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)

@app.route("/health")
def health():
    return jsonify({
        "cnn_ready": CNN_MODEL is not None,
        "lstm_ready": LSTM_MODEL is not None,
        "cnn_classes": CNN_CLASSES or [],
        "lstm_classes": LSTM_CLASSES or []
    })

# 1. Endpoint CNN (Gambar Statis)
@app.route("/predict", methods=["POST"])
def predict_static():
    if not CNN_MODEL: return jsonify({"error": "CNN belum siap"}), 503
    try:
        data = request.json
        img_bytes = safe_b64_to_bytes(data['image'])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess CNN (224x224)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CNN_IMG_SIZE, CNN_IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        
        label, conf = predict_cnn(CNN_MODEL, CNN_CLASSES, img)
        return jsonify({"label": label, "confidence": conf})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# 2. Endpoint LSTM (Video Gerakan)
@app.route("/predict_lstm", methods=["POST"])
def predict_dynamic():
    if not LSTM_MODEL: return jsonify({"error": "LSTM belum siap (Training dlu)"}), 503
    temp_path = None
    try:
        data = request.json
        video_bytes = safe_b64_to_bytes(data['video'])
        
        # Simpan video sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=VIDEOS_DIR) as tmp:
            tmp.write(video_bytes)
            temp_path = tmp.name
            
        # Extract 20 frames
        cap = cv2.VideoCapture(temp_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            indices = np.linspace(0, total_frames-1, LSTM_FRAMES, dtype=int)
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Preprocess LSTM (128x128)
                    frame = cv2.resize(frame, (LSTM_IMG_SIZE, LSTM_IMG_SIZE))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
        cap.release()
        
        # Padding jika kurang
        while len(frames) < LSTM_FRAMES:
            frames.append(np.zeros((LSTM_IMG_SIZE, LSTM_IMG_SIZE, 3), dtype=np.uint8))
            
        # Convert to batch (1, 20, 128, 128, 3)
        frames_arr = np.array(frames, dtype="float32") / 255.0
        frames_arr = np.expand_dims(frames_arr, axis=0)
        
        label, conf = predict_lstm(LSTM_MODEL, LSTM_CLASSES, frames_arr)
        return jsonify({"label": label, "confidence": conf})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path) 
            except: pass

if __name__ == "__main__":
    ensure_dirs()
    load_models()
    app.run(port=5000, debug=False)