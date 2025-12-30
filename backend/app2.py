import os
import time
import json
import base64
import traceback
import tempfile
import threading
from pathlib import Path
from datetime import datetime

# ============================================
# 🛑 CONFIG SANGAT PENTING (LEGACY MODE)
# ============================================
# Kita paksa pakai Mode Legacy biar CNN Teachable Machine jalan
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp # ✅ Tambahan untuk LSTM CSV
from werkzeug.exceptions import NotFound

# ============================================
# 🔧 IMPORT KHUSUS (TF_KERAS)
# ============================================
try:
    import tf_keras as keras
    from tf_keras.models import load_model
    from tf_keras.layers import DepthwiseConv2D, InputLayer
    print("✅ Berhasil memuat tf_keras (Mode Kompatibilitas Aman)")
except ImportError:
    print("❌ GAWAT! Library 'tf_keras' belum diinstall.")
    raise ImportError("Wajib install: pip install tf_keras")

# ============================================
# 🔧 PATCH FIX (PENYELAMAT MODEL)
# ============================================

# 1. Patch untuk CNN (Teachable Machine)
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None) 
        super().__init__(**kwargs)

# 2. Patch untuk LSTM
class CustomInputLayer(InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)

# =========================
# 📁 Path & Konfigurasi
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = BASE_DIR / "temp_videos"
LOGS_DIR = BASE_DIR / "logs"

# File Model CNN (TETAP)
CNN_MODEL_FILE = MODELS_DIR / "keras_model.h5"
CNN_LABELS_FILE = MODELS_DIR / "labels.txt"

# File Model LSTM (UBAH KE MODEL CSV)
# Pastikan nama filenya sesuai hasil training train_lstm_csv.py
LSTM_MODEL_FILE = MODELS_DIR / "bisindo_lstm_csv.h5" 
LSTM_LABELS_FILE = MODELS_DIR / "class_names_lstm.json"

FRONTEND_DIR = BASE_DIR.parent / "frontend"
LOG_FILE = LOGS_DIR / "server.log"

# Konfigurasi Dimensi
IMG_HEIGHT = 224      # CNN
IMG_WIDTH = 224       # CNN
LSTM_FRAMES = 20      # LSTM

# Konfigurasi MediaPipe (Sesuai prepo2.py)
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
TOTAL_FEATURES = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK # 126

ENABLE_AUTO_RELOAD = os.environ.get("AUTO_RELOAD_MODELS", "0") == "1"
MODEL_POLL_INTERVAL = 10

# =========================
# 🧠 Global Models & MediaPipe
# =========================
CNN_MODEL = None
CLASS_NAMES = None 

LSTM_MODEL = None
LSTM_CLASSES = None 

# Init MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

_models_lock = threading.Lock()
_model_load_time = 0

# =========================
# ⚙️ Utilities
# =========================
def ensure_dirs():
    for d in [MODELS_DIR, DATA_DIR, VIDEOS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

def log_event(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# =========================
# 🖼️ Preprocessing CNN
# =========================
def preprocess_teachable_machine(bgr_img):
    if bgr_img is None or bgr_img.size == 0: raise ValueError("Invalid image")
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)

# =========================
# 🖐️ Preprocessing LSTM (Landmark Extraction)
# =========================
def extract_landmarks(frame):
    """
    Ekstrak 126 fitur (x,y,z) dari frame menggunakan MediaPipe
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Wadah data (flat array 126 angka)
    frame_data = np.zeros(TOTAL_FEATURES)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= NUM_HANDS: break
            for j, lm in enumerate(hand_landmarks.landmark):
                idx = (i * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK) + (j * COORDS_PER_LANDMARK)
                frame_data[idx]     = lm.x
                frame_data[idx + 1] = lm.y
                frame_data[idx + 2] = lm.z
                
    return frame_data

# =========================
# 🔄 Load Models
# =========================
def load_all_models():
    global CNN_MODEL, CLASS_NAMES, LSTM_MODEL, LSTM_CLASSES, _model_load_time
    with _models_lock:
        log_event("🔄 Memulai proses pemuatan model...")
        
        # --- 1. Load CNN (TEACHABLE MACHINE - JANGAN UBAH) ---
        try:
            if CNN_MODEL_FILE.exists() and CNN_LABELS_FILE.exists():
                log_event(f"  ↳ Memuat CNN dari {CNN_MODEL_FILE.name}...")
                with open(CNN_LABELS_FILE, "r") as f:
                    raw_lines = f.readlines()
                    CLASS_NAMES = []
                    for line in raw_lines:
                        clean_line = line.strip()
                        if len(clean_line) > 2 and clean_line[0].isdigit():
                            CLASS_NAMES.append(clean_line[2:].strip())
                        else:
                            CLASS_NAMES.append(clean_line)

                CNN_MODEL = load_model(
                    str(CNN_MODEL_FILE), 
                    custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
                    compile=False
                )
                log_event(f"  ✅ CNN Loaded. Classes: {CLASS_NAMES}")
            else:
                log_event(f"  ❌ CNN Files Missing")
        except Exception as e:
            log_event(f"  ❌ Error loading CNN: {e}")

        # --- 2. Load LSTM (MODEL BARU - CSV BASED) ---
        try:
            if LSTM_MODEL_FILE.exists() and LSTM_LABELS_FILE.exists():
                log_event(f"  ↳ Memuat LSTM dari {LSTM_MODEL_FILE.name}...")
                
                with open(LSTM_LABELS_FILE, "r") as f:
                    LSTM_CLASSES = json.load(f)

                # Load Model (Patch InputLayer tetap berguna jika ada metadata lama)
                LSTM_MODEL = load_model(
                    str(LSTM_MODEL_FILE),
                    custom_objects={'InputLayer': CustomInputLayer}, 
                    compile=False
                )
                log_event(f"  ✅ LSTM Loaded. Classes: {LSTM_CLASSES}")
            else:
                log_event(f"  ⚠️ LSTM Files Missing (Cek bisindo_lstm_csv.h5)")
        except Exception as e:
            log_event(f"  ❌ Error loading LSTM: {e}")
            traceback.print_exc()

        _model_load_time = time.time()

# =========================
# 👁️ Poller
# =========================
def model_poller(stop_event):
    if not ENABLE_AUTO_RELOAD: return
    last_time = 0
    while not stop_event.is_set():
        time.sleep(MODEL_POLL_INTERVAL)
        try:
            if not CNN_MODEL_FILE.exists(): continue
            cur_time = os.path.getmtime(CNN_MODEL_FILE)
            if last_time == 0: last_time = cur_time
            elif cur_time > last_time:
                log_event("👁️ Model update detected...")
                load_all_models()
                last_time = cur_time
        except Exception: continue

# =========================
# 🧩 Helper Base64
# =========================
def safe_b64_to_bytes(data_b64: str) -> bytes:
    if "base64," in data_b64: data_b64 = data_b64.split("base64,")[1]
    padding = len(data_b64) % 4
    if padding != 0: data_b64 += "=" * (4 - padding)
    return base64.b64decode(data_b64)

# =========================
# 🚀 Flask App & Routes
# =========================
app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_error(e):
    if isinstance(e, NotFound): return jsonify({"error": "Not found"}), 404
    log_event(f"❌ Unhandled error: {e}")
    return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    try: return send_from_directory(str(FRONTEND_DIR), 'index.html')
    except Exception: return jsonify({"status": "Backend Running"}), 200

@app.route("/<path:filename>")
def serve_static(filename):
    try: return send_from_directory(str(FRONTEND_DIR), filename)
    except FileNotFoundError: return jsonify({"error": "File not found"}), 404

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "cnn_ready": bool(CNN_MODEL is not None),
        "lstm_ready": bool(LSTM_MODEL is not None),
        "uptime": float(time.time() - _model_load_time) if _model_load_time else 0
    })

# =========================
# 🧠 CNN Predict (GAMBAR/FRAME)
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if CNN_MODEL is None: return jsonify({"error": "Model CNN belum dimuat"}), 503
        data = request.json
        raw = safe_b64_to_bytes(data.get("image", ""))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        
        inp = preprocess_teachable_machine(img)
        pred = CNN_MODEL.predict(inp, verbose=0)
        idx = np.argmax(pred)
        label = CLASS_NAMES[idx] if CLASS_NAMES else f"{idx}"
        conf = float(pred[0][idx])
        
        if np.random.random() < 0.1: log_event(f"🔮 CNN: {label} ({conf:.2f})")
        return jsonify({"label": label, "confidence": conf})
    except Exception as e:
        log_event(f"❌ Predict Error: {e}")
        return jsonify({"error": str(e)}), 500

# =========================
# 🎥 CNN Video (FRAME-BY-FRAME)
# =========================
@app.route("/predict_video", methods=["POST"])
def predict_video():
    temp_path = None
    try:
        if CNN_MODEL is None: return jsonify({"error": "Model CNN belum dimuat"}), 503
        data = request.json
        raw = safe_b64_to_bytes(data.get("video", ""))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=str(VIDEOS_DIR)) as tmp:
            tmp.write(raw)
            temp_path = tmp.name
        
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return jsonify({"error": "Video kosong"}), 400
        
        indices = np.linspace(0, total_frames - 1, min(total_frames, 30), dtype=int)
        preds = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret: continue
            try:
                inp = preprocess_teachable_machine(frame)
                preds.append(CNN_MODEL.predict(inp, verbose=0)[0])
            except: continue
        cap.release()
        
        if not preds: return jsonify({"error": "No frames"}), 400
        avg = np.mean(preds, axis=0)
        idx = int(np.argmax(avg))
        label = CLASS_NAMES[idx] if CLASS_NAMES else f"{idx}"
        conf = float(avg[idx])
        
        return jsonify({"label": label, "confidence": conf, "mode": "CNN_VIDEO"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

# =========================
# 🌊 LSTM Predict (CSV/LANDMARK BASED)
# =========================
@app.route("/predict_lstm", methods=["POST"])
def predict_dynamic():
    if not LSTM_MODEL: return jsonify({"error": "Model LSTM belum dimuat"}), 503
    temp_path = None
    try:
        data = request.json
        raw = safe_b64_to_bytes(data.get("video", ""))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=str(VIDEOS_DIR)) as tmp:
            tmp.write(raw)
            temp_path = tmp.name
            
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        extracted_sequence = []
        
        if total_frames > 0:
            # Sampling 20 Frame rata
            indices = np.linspace(0, total_frames-1, LSTM_FRAMES, dtype=int)
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # 🚀 EXTRACTION: Ubah Frame -> 126 Landmark Features
                    landmarks = extract_landmarks(frame)
                    extracted_sequence.append(landmarks)
        
        cap.release()
        
        # Validation & Padding
        if len(extracted_sequence) == 0:
            return jsonify({"error": "Gagal ekstrak fitur dari video"}), 400
            
        while len(extracted_sequence) < LSTM_FRAMES:
            # Padding dengan copy frame terakhir (atau nol)
            extracted_sequence.append(extracted_sequence[-1])
            
        # Convert ke Numpy Array (1, 20, 126)
        # Input bukan lagi gambar (20, 128, 128, 3), tapi array angka
        sequence_arr = np.array(extracted_sequence, dtype=np.float32)
        sequence_arr = np.expand_dims(sequence_arr, axis=0)
        
        # PREDIKSI
        pred = LSTM_MODEL.predict(sequence_arr, verbose=0)
        idx = int(np.argmax(pred[0]))
        conf = float(pred[0][idx])
        label = LSTM_CLASSES[idx] if LSTM_CLASSES else f"Class {idx}"
        
        log_event(f"🎬 LSTM (Landmark): {label} ({conf:.2f})")
        return jsonify({"label": label, "confidence": conf})
        
    except Exception as e:
        log_event(f"❌ LSTM Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)

@app.route("/reload_models", methods=["POST"])
def reload_models():
    load_all_models()
    return jsonify({"message": "Reloaded", "cnn": bool(CNN_MODEL), "lstm": bool(LSTM_MODEL)})

if __name__ == "__main__":
    ensure_dirs()
    log_event("🚀 Server Starting (Mode: CNN Legacy + LSTM CSV)...")
    load_all_models()
    
    stop_event = threading.Event()
    poller = threading.Thread(target=model_poller, args=(stop_event,), daemon=True)
    poller.start()
    
    try: app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)
    except KeyboardInterrupt: pass
    finally:
        stop_event.set()
        poller.join(timeout=2)