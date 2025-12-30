"""
model_utils.py
Helper functions untuk loading dan prediksi model
- CNN: Menggunakan format Teachable Machine (.h5 + .txt)
- LSTM: Menggunakan format training custom (.keras + .json)
"""

import json
import numpy as np
from pathlib import Path
from tensorflow import keras
import traceback
import os

# =========================
# 📁 Paths Default
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Tentukan nama file secara eksplisit biar tidak bingung
# 1. CNN (Teachable Machine)
CNN_MODEL_FILE = MODELS_DIR / "keras_model.h5"
CNN_LABELS_FILE = MODELS_DIR / "labels.txt"

# 2. LSTM (Hasil train_lstm.py)
LSTM_MODEL_FILE = MODELS_DIR / "bisindo_lstm_v2.h5"
LSTM_LABELS_FILE = MODELS_DIR / "class_names_lstm.json"

# ==========================================
# 🧠 SECTION 1: CNN MODEL (Teachable Machine)
# ==========================================

def load_cnn_model():
    """
    Load CNN model dari file Teachable Machine (h5 & txt)
    """
    try:
        # Cek keberadaan file
        if not CNN_MODEL_FILE.exists():
            print(f"❌ CNN Error: File '{CNN_MODEL_FILE.name}' tidak ditemukan di folder models!")
            return None, None
        
        if not CNN_LABELS_FILE.exists():
            print(f"❌ CNN Error: File '{CNN_LABELS_FILE.name}' tidak ditemukan di folder models!")
            return None, None

        # 1. Load Model .h5
        # compile=False penting untuk model hasil export TM agar tidak error optimizer
        print(f"📥 Loading CNN (Teachable Machine) dari: {CNN_MODEL_FILE.name}...")
        model = keras.models.load_model(str(CNN_MODEL_FILE), compile=False)
        
        # 2. Load Labels .txt
        # Teachable Machine sering kasih format "0 A", "1 B". Kita perlu bersihkan.
        class_names = []
        with open(CNN_LABELS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                clean_line = line.strip()
                # Jika formatnya "0 NamaKelas", ambil nama kelasnya saja
                parts = clean_line.split(" ", 1)
                if len(parts) > 1 and parts[0].isdigit():
                    class_names.append(parts[1])
                else:
                    class_names.append(clean_line)
        
        print(f"✅ CNN READY: {len(class_names)} classes dimuat ({class_names[:3]}...)")
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")
        traceback.print_exc()
        return None, None

def predict_cnn(model, class_names, input_img):
    """
    Prediksi menggunakan CNN
    """
    if model is None or class_names is None:
        raise ValueError("Model CNN atau class names belum dimuat")
    
    # Predict
    predictions = model.predict(input_img, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_idx])
    
    # Get label
    if pred_idx < len(class_names):
        label = class_names[pred_idx]
    else:
        label = f"Class {pred_idx}"
    
    return label, confidence


# ==========================================
# 🧠 SECTION 2: LSTM MODEL (Custom Training)
# ==========================================

def load_lstm_model():
    """
    Load LSTM model dari hasil train_lstm.py
    Target: bisindo_lstm_v2.keras & class_names_lstm.json
    """
    try:
        # Cek keberadaan file
        if not LSTM_MODEL_FILE.exists():
            # Silent fail (return None) biar server tetap jalan kalau belum training LSTM
            print(f"ℹ️ LSTM Info: File '{LSTM_MODEL_FILE.name}' belum ada (Skip LSTM).")
            return None, None
        
        if not LSTM_LABELS_FILE.exists():
            print(f"⚠️ LSTM Warning: Model ada tapi '{LSTM_LABELS_FILE.name}' hilang.")
            return None, None
        
        # 1. Load Model .keras
        print(f"📥 Loading LSTM Model dari: {LSTM_MODEL_FILE.name}...")
        model = keras.models.load_model(str(LSTM_MODEL_FILE))
        
        # 2. Load Labels .json
        with open(LSTM_LABELS_FILE, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        
        print(f"✅ LSTM READY: {len(class_names)} classes dimuat.")
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")
        return None, None

def predict_lstm(model, class_names, frames_array):
    if model is None:
        raise ValueError("Model LSTM belum dimuat")
    
    # Prediksi
    predictions = model.predict(frames_array, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_idx])
    
    # Ambil Label
    if class_names and pred_idx < len(class_names):
        label = class_names[pred_idx]
    else:
        label = f"Class {pred_idx}"
    
    return label, confidence

# =========================
# 🧪 Test Functions
# =========================
def test_models():
    print("="*60)
    print("🧪 Testing Model Loading...")
    print("="*60)
    
    # Test CNN
    cnn_model, cnn_classes = load_cnn_model()
    if cnn_model:
        print(f"✅ CNN OK. Path: {CNN_MODEL_FILE.name}")
    else:
        print(f"❌ CNN GAGAL. Pastikan {CNN_MODEL_FILE.name} ada di folder models.")
    
    print("-" * 30)

    # Test LSTM
    lstm_model, lstm_classes = load_lstm_model()
    if lstm_model:
        print(f"✅ LSTM OK. Path: {LSTM_MODEL_FILE.name}")
    else:
        print(f"ℹ️ LSTM Belum tersedia (Jalankan train_lstm.py dulu).")
    
    print("="*60)

if __name__ == "__main__":
    test_models()