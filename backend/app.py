import os
import time
import json
import base64
import traceback
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta

<<<<<<< HEAD
# ============================================
# 🛑 CONFIG SANGAT PENTING (LEGACY MODE)
# ============================================
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, jsonify, send_from_directory, redirect, render_template_string
from flask_cors import CORS
import numpy as np
import cv2
import mediapipe as mp
from werkzeug.exceptions import NotFound
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
=======
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
from werkzeug.exceptions import NotFound
>>>>>>> 9c67073f5b774a3eb292f7627bc7bfd6abe18d66

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
    print("Install dengan: pip install tf_keras tensorflow mediapipe opencv-python flask flask-cors pyjwt")
    raise ImportError("Wajib install: pip install tf_keras")

# ============================================
# 🔧 PATCH FIX
# ============================================
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None) 
        super().__init__(**kwargs)

class CustomInputLayer(InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(**kwargs)

# =========================
# 📂 Path & Konfigurasi
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = BASE_DIR / "temp_videos"
LOGS_DIR = BASE_DIR / "logs"

<<<<<<< HEAD
CNN_MODEL_FILE = MODELS_DIR / "keras_model.h5"
CNN_LABELS_FILE = MODELS_DIR / "labels.txt"
LSTM_MODEL_FILE = MODELS_DIR / "bisindo_lstm_csv.h5" 
LSTM_LABELS_FILE = MODELS_DIR / "class_names_lstm.json"
=======
# Frontend directory (sibling folder)
FRONTEND_DIR = BASE_DIR.parent / "frontend"

LOG_FILE = LOGS_DIR / "server.log"
>>>>>>> 9c67073f5b774a3eb292f7627bc7bfd6abe18d66

LOG_FILE = LOGS_DIR / "server.log"
USERS_FILE = DATA_DIR / "users.json"

# JWT Config
SECRET_KEY = "signtara_secret_key_2024_demo"
TOKEN_EXPIRY_HOURS = 24

# Konfigurasi Dimensi
IMG_HEIGHT = 224
IMG_WIDTH = 224
LSTM_FRAMES = 20

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
TOTAL_FEATURES = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK

# =========================
# 🧠 Global Models & MediaPipe
# =========================
CNN_MODEL = None
CLASS_NAMES = None 
LSTM_MODEL = None
LSTM_CLASSES = None 

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
    
    # Create dummy users file if not exists
    if not USERS_FILE.exists():
        dummy_users = {
            "demo": {
                "email": "demo@signtara.com",
                "password": generate_password_hash("demo123")
            }
        }
        with open(USERS_FILE, "w") as f:
            json.dump(dummy_users, f, indent=2)

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
# 🔐 Auth Functions
# =========================
def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def create_token(username):
    payload = {
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["username"]
    except:
        return None

# =========================
# 🖼️ Preprocessing CNN
# =========================
def preprocess_teachable_machine(bgr_img):
    if bgr_img is None or bgr_img.size == 0: 
        raise ValueError("Invalid image")
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0)

# =========================
# 🖐️ Preprocessing LSTM
# =========================
def extract_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame_data = np.zeros(TOTAL_FEATURES)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= NUM_HANDS: break
            for j, lm in enumerate(hand_landmarks.landmark):
                idx = (i * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK) + (j * COORDS_PER_LANDMARK)
                frame_data[idx] = lm.x
                frame_data[idx + 1] = lm.y
                frame_data[idx + 2] = lm.z
                
    return frame_data

# =========================
# 📄 Load Models
# =========================
def load_all_models():
    global CNN_MODEL, CLASS_NAMES, LSTM_MODEL, LSTM_CLASSES, _model_load_time
    with _models_lock:
        log_event("🔄 Memulai proses pemuatan model...")
        
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
                log_event(f"  ⚠️ CNN Files Missing (Optional)")
        except Exception as e:
            log_event(f"  ⚠️ CNN loading skipped: {e}")

        try:
            if LSTM_MODEL_FILE.exists() and LSTM_LABELS_FILE.exists():
                log_event(f"  ↳ Memuat LSTM dari {LSTM_MODEL_FILE.name}...")
                with open(LSTM_LABELS_FILE, "r") as f:
                    LSTM_CLASSES = json.load(f)

                LSTM_MODEL = load_model(
                    str(LSTM_MODEL_FILE),
                    custom_objects={'InputLayer': CustomInputLayer}, 
                    compile=False
                )
                log_event(f"  ✅ LSTM Loaded. Classes: {LSTM_CLASSES}")
            else:
                log_event(f"  ⚠️ LSTM Files Missing (Optional)")
        except Exception as e:
            log_event(f"  ⚠️ LSTM loading skipped: {e}")

        _model_load_time = time.time()

# =========================
# 🧩 Helper Base64
# =========================
def safe_b64_to_bytes(data_b64: str) -> bytes:
    if "base64," in data_b64: 
        data_b64 = data_b64.split("base64,")[1]
    padding = len(data_b64) % 4
    if padding != 0: 
        data_b64 += "=" * (4 - padding)
    return base64.b64decode(data_b64)

# =========================
# 🚀 Flask App
# =========================
app = Flask(__name__)
CORS(app)

<<<<<<< HEAD
# =========================
# 🌐 ROUTES - PUBLIC PAGES
# =========================
@app.route("/")
def home():
    return render_template_string(HOME_HTML)

@app.route("/login")
def login_page():
    return render_template_string(LOGIN_HTML)

@app.route("/register")
def register_page():
    return render_template_string(REGISTER_HTML)

@app.route("/index")
def index_page():
    return render_template_string(INDEX_HTML)

@app.route("/premium")
def premium_page():
    return render_template_string(PREMIUM_HTML)
=======
# Error handler - Handle 404 gracefully
@app.errorhandler(404)
def handle_404(e):
    """Handle 404 errors without logging as critical errors"""
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(Exception)
def handle_error(e):
    """Handle other errors"""
    # Skip logging for 404 errors (already handled above)
    if isinstance(e, NotFound):
        return jsonify({"error": "Not found"}), 404
    
    log_event(f"❌ Unhandled error: {e}")
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

# =========================
# 🌐 Serve Frontend
# =========================
@app.route("/")
def index():
    """Serve the main HTML file from frontend folder"""
    try:
        if not FRONTEND_DIR.exists():
            log_event(f"❌ Frontend folder tidak ditemukan: {FRONTEND_DIR}")
            return jsonify({
                "error": "Frontend folder tidak ditemukan",
                "expected_path": str(FRONTEND_DIR),
                "tip": "Pastikan folder 'frontend' ada di parent directory"
            }), 404
        
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            log_event(f"❌ index.html tidak ditemukan di: {index_path}")
            return jsonify({
                "error": "index.html tidak ditemukan",
                "expected_path": str(index_path)
            }), 404
        
        return send_from_directory(str(FRONTEND_DIR), 'index.html')
    except Exception as e:
        log_event(f"❌ Error serving index.html: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/<path:filename>")
def serve_static(filename):
    """Serve static files (CSS, JS, images) from frontend folder"""
    try:
        return send_from_directory(str(FRONTEND_DIR), filename)
    except FileNotFoundError:
        return jsonify({"error": f"File {filename} tidak ditemukan"}), 404

# =========================
# 📊 API Status
# =========================
@app.route("/api/status")
def api_status():
    """API status endpoint"""
    classes = []
    if CLASS_NAMES is not None:
        classes = [str(c) for c in CLASS_NAMES]
    
    return jsonify({
        "status": "ok",
        "cnn_loaded": bool(CNN_MODEL is not None),
        "classes": classes,
        "version": "5.0 CNN-Only",
        "uptime": float(time.time() - _model_load_time) if _model_load_time else 0,
        "frontend_path": str(FRONTEND_DIR)
    })
>>>>>>> 9c67073f5b774a3eb292f7627bc7bfd6abe18d66

# =========================
# 🔐 AUTH API ROUTES
# =========================
@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.json
        users = load_users()
        
        if data["username"] in users:
            return jsonify({"error": "Username sudah digunakan"}), 400
        
        users[data["username"]] = {
            "email": data["email"],
            "password": generate_password_hash(data["password"])
        }
        save_users(users)
        
        log_event(f"✅ Registrasi: {data['username']}")
        return jsonify({"message": "Registrasi berhasil"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.json
        users = load_users()
        
        if data["username"] not in users:
            return jsonify({"error": "Username tidak ditemukan"}), 401
        
        user = users[data["username"]]
        if not check_password_hash(user["password"], data["password"]):
            return jsonify({"error": "Password salah"}), 401
        
        token = create_token(data["username"])
        log_event(f"✅ Login: {data['username']}")
        return jsonify({"access_token": token}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# 🧠 ML API ROUTES
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "cnn_ready": bool(CNN_MODEL is not None),
        "lstm_ready": bool(LSTM_MODEL is not None),
        "uptime": float(time.time() - _model_load_time) if _model_load_time else 0
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if CNN_MODEL is None: 
            return jsonify({"error": "Model CNN belum dimuat"}), 503
        data = request.json
        raw = safe_b64_to_bytes(data.get("image", ""))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        
        inp = preprocess_teachable_machine(img)
        pred = CNN_MODEL.predict(inp, verbose=0)
        idx = np.argmax(pred)
        label = CLASS_NAMES[idx] if CLASS_NAMES else f"{idx}"
        conf = float(pred[0][idx])
        
        return jsonify({"label": label, "confidence": conf})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_video", methods=["POST"])
def predict_video():
    temp_path = None
    try:
        if CNN_MODEL is None: 
            return jsonify({"error": "Model CNN belum dimuat"}), 503
        data = request.json
        raw = safe_b64_to_bytes(data.get("video", ""))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=str(VIDEOS_DIR)) as tmp:
            tmp.write(raw)
            temp_path = tmp.name
        
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: 
            return jsonify({"error": "Video kosong"}), 400
        
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
        
        if not preds: 
            return jsonify({"error": "No frames"}), 400
        avg = np.mean(preds, axis=0)
        idx = int(np.argmax(avg))
        label = CLASS_NAMES[idx] if CLASS_NAMES else f"{idx}"
        conf = float(avg[idx])
        
        return jsonify({"label": label, "confidence": conf, "mode": "CNN_VIDEO"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path): 
            os.remove(temp_path)

@app.route("/predict_lstm", methods=["POST"])
def predict_dynamic():
    if not LSTM_MODEL: 
        return jsonify({"error": "Model LSTM belum dimuat"}), 503
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
            indices = np.linspace(0, total_frames-1, LSTM_FRAMES, dtype=int)
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    landmarks = extract_landmarks(frame)
                    extracted_sequence.append(landmarks)
        
        cap.release()
        
        if len(extracted_sequence) == 0:
            return jsonify({"error": "Gagal ekstrak fitur dari video"}), 400
            
        while len(extracted_sequence) < LSTM_FRAMES:
            extracted_sequence.append(extracted_sequence[-1])
            
        sequence_arr = np.array(extracted_sequence, dtype=np.float32)
        sequence_arr = np.expand_dims(sequence_arr, axis=0)
        
        pred = LSTM_MODEL.predict(sequence_arr, verbose=0)
        idx = int(np.argmax(pred[0]))
        conf = float(pred[0][idx])
        label = LSTM_CLASSES[idx] if LSTM_CLASSES else f"Class {idx}"
        
        return jsonify({"label": label, "confidence": conf})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path): 
            os.remove(temp_path)

# =========================
# 📄 HTML TEMPLATES
# =========================
HOME_HTML = '''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signtara - Jembatan Komunikasi Tanpa Batas</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .blob {position:absolute;width:500px;height:500px;border-radius:50%;filter:blur(100px);opacity:0.3;animation:float 20s infinite ease-in-out}
        .blob-1{background:linear-gradient(45deg,#4F46E5,#7C3AED);top:10%;left:10%;animation-delay:0s}
        .blob-2{background:linear-gradient(45deg,#EC4899,#F59E0B);bottom:10%;right:10%;animation-delay:5s}
        @keyframes float{0%,100%{transform:translateY(0) scale(1)}50%{transform:translateY(-50px) scale(1.1)}}
    </style>
</head>
<body class="bg-gray-50 min-h-screen text-gray-900 overflow-x-hidden">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    
<<<<<<< HEAD
    <nav class="fixed w-full z-50 bg-white/80 backdrop-blur-md shadow-sm">
        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <span class="text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600">Signtara</span>
            <div class="flex gap-4">
                <a href="/login" class="text-indigo-600 font-bold hover:underline">Masuk</a>
                <a href="/register" class="px-5 py-2.5 rounded-full bg-gray-900 text-white font-bold hover:bg-gray-800 transition shadow-lg">Daftar</a>
            </div>
        </div>
    </nav>

    <header class="relative pt-32 pb-20 px-6">
        <div class="max-w-7xl mx-auto text-center relative z-10">
            <span class="inline-block py-1 px-3 rounded-full bg-indigo-100 text-indigo-700 text-sm font-bold mb-6">🚀 Teknologi AI untuk Inklusivitas</span>
            <h1 class="text-5xl md:text-7xl font-extrabold leading-tight mb-6">
                Bicara Tanpa Suara,<br>
                <span class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600">Dimengerti Seketika.</span>
            </h1>
            <p class="text-lg text-gray-600 mb-10 max-w-2xl mx-auto">
                Aplikasi penerjemah BISINDO real-time bertenaga AI. Membuka dunia komunikasi bagi Teman Tuli dan Teman Dengar.
            </p>
            <a href="/register" class="inline-block px-8 py-4 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold text-lg shadow-xl hover:shadow-indigo-500/40 hover:-translate-y-1 transition">
                Mulai Gratis →
            </a>
        </div>
    </header>

    <section class="py-20 px-6 relative z-10">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-3xl md:text-4xl font-bold mb-4">Kenapa Signtara?</h2>
                <p class="text-gray-500">Teknologi canggih yang dikemas sederhana</p>
            </div>

            <div class="grid md:grid-cols-3 gap-8">
                <div class="bg-white/60 backdrop-blur-sm rounded-2xl p-8 hover:-translate-y-2 transition shadow-lg">
                    <div class="w-14 h-14 rounded-2xl bg-indigo-100 flex items-center justify-center text-indigo-600 text-3xl mb-6">🎥</div>
                    <h3 class="text-xl font-bold mb-3">Deteksi Real-time</h3>
                    <p class="text-gray-600">Terjemahkan gerakan tangan menjadi teks secara instan tanpa delay.</p>
                </div>

                <div class="bg-white/60 backdrop-blur-sm rounded-2xl p-8 hover:-translate-y-2 transition shadow-lg">
                    <div class="w-14 h-14 rounded-2xl bg-purple-100 flex items-center justify-center text-purple-600 text-3xl mb-6">🧠</div>
                    <h3 class="text-xl font-bold mb-3">Kecerdasan Buatan</h3>
                    <p class="text-gray-600">Didukung model Deep Learning (LSTM & CNN) yang terus belajar.</p>
                </div>

                <div class="bg-white/60 backdrop-blur-sm rounded-2xl p-8 hover:-translate-y-2 transition shadow-lg">
                    <div class="w-14 h-14 rounded-2xl bg-emerald-100 flex items-center justify-center text-emerald-600 text-3xl mb-6">📚</div>
                    <h3 class="text-xl font-bold mb-3">Kamus & Edukasi</h3>
                    <p class="text-gray-600">Bukan hanya menerjemahkan, tapi juga membantu Anda belajar.</p>
                </div>
            </div>
        </div>
    </section>

    <footer class="py-8 text-center text-gray-500">
        © 2024 Signtara Project. Demo Version.
    </footer>
</body>
</html>'''

LOGIN_HTML = '''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Masuk - Signtara</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .blob {position:absolute;width:500px;height:500px;border-radius:50%;filter:blur(100px);opacity:0.3;animation:float 20s infinite}
        .blob-1{background:linear-gradient(45deg,#4F46E5,#7C3AED);top:10%;left:10%}
        .blob-2{background:linear-gradient(45deg,#EC4899,#F59E0B);bottom:10%;right:10%;animation-delay:5s}
        @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-50px)}}
    </style>
</head>
<body class="min-h-screen flex items-center justify-center bg-gray-50 relative overflow-hidden">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>

    <div class="bg-white/80 backdrop-blur-md rounded-3xl p-8 w-full max-w-md mx-4 relative z-10 shadow-2xl">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-bold mb-2">Selamat Datang</h1>
            <p class="text-gray-500">Masuk untuk mulai berkomunikasi</p>
        </div>

        <form id="loginForm" class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Username</label>
                <input type="text" id="username" class="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-indigo-600 focus:outline-none transition" placeholder="Masukkan username" required>
            </div>
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Password</label>
                <input type="password" id="password" class="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-indigo-600 focus:outline-none transition" placeholder="••••••••" required>
            </div>

            <div id="errorMessage" class="hidden text-sm text-red-600 bg-red-50 p-3 rounded-lg"></div>

            <button type="submit" id="submitButton" class="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold hover:shadow-xl transition">Masuk</button>
        </form>

        <p class="mt-6 text-center text-sm text-gray-600">
            Belum punya akun? 
            <a href="/register" class="text-indigo-600 font-bold hover:underline">Daftar di sini</a>
        </p>
        
        <div class="mt-4 p-3 bg-blue-50 rounded-lg text-xs text-blue-700">
            <strong>Demo:</strong> Username: <code class="bg-blue-100 px-1 rounded">demo</code>, Password: <code class="bg-blue-100 px-1 rounded">demo123</code>
        </div>
    </div>

    <script>
        const loginForm = document.getElementById('loginForm');
        const errorMessage = document.getElementById('errorMessage');
        const submitButton = document.getElementById('submitButton');

        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            errorMessage.classList.add('hidden');
            submitButton.disabled = true;
            submitButton.textContent = 'Memproses...';

            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });

                const data = await response.json();

                if (!response.ok) throw new Error(data.error || 'Gagal Login');

                localStorage.setItem('signtara_token', data.access_token);
                window.location.href = '/index';

            } catch (error) {
                errorMessage.textContent = error.message;
                errorMessage.classList.remove('hidden');
                submitButton.disabled = false;
                submitButton.textContent = 'Masuk';
            }
        });
    </script>
</body>
</html>'''

REGISTER_HTML = '''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daftar - Signtara</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .blob {position:absolute;width:500px;height:500px;border-radius:50%;filter:blur(100px);opacity:0.3;animation:float 20s infinite}
        .blob-1{background:linear-gradient(45deg,#4F46E5,#7C3AED);top:10%;left:10%}
        .blob-2{background:linear-gradient(45deg,#EC4899,#F59E0B);bottom:10%;right:10%;animation-delay:5s}
        @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-50px)}}
    </style>
</head>
<body class="min-h-screen flex items-center justify-center bg-gray-50 relative overflow-hidden">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>

    <div class="bg-white/80 backdrop-blur-md rounded-3xl p-8 w-full max-w-md mx-4 relative z-10 shadow-2xl">
        <div class="text-center mb-8">
            <h1 class="text-3xl font-bold mb-2">Buat Akun</h1>
            <p class="text-gray-500">Bergabung dengan komunitas Signtara</p>
        </div>

        <form id="registerForm" class="space-y-4">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input type="email" id="email" class="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-indigo-600 focus:outline-none transition" required>
            </div>
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <input type="text" id="username" class="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-indigo-600 focus:outline-none transition" required>
            </div>
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Password</label>
                <input type="password" id="password" class="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-indigo-600 focus:outline-none transition" required>
            </div>

            <div id="message" class="hidden p-3 rounded-lg text-sm"></div>

            <button type="submit" id="submitButton" class="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold hover:shadow-xl transition">Daftar Akun</button>
        </form>

        <p class="mt-6 text-center text-sm text-gray-600">
            Sudah punya akun? 
            <a href="/login" class="text-indigo-600 font-bold hover:underline">Masuk</a>
        </p>
    </div>

    <script>
        document.getElementById('registerForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const btn = document.getElementById('submitButton');
            const msg = document.getElementById('message');
            btn.disabled = true;
            btn.textContent = "Mendaftar...";

            try {
                const res = await fetch('/api/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        email: document.getElementById('email').value,
                        username: document.getElementById('username').value,
                        password: document.getElementById('password').value
                    })
                });
                const data = await res.json();
                if(!res.ok) throw new Error(data.error);

                msg.className = "text-green-700 bg-green-100 p-3 rounded-lg block";
                msg.textContent = "Berhasil! Mengalihkan ke login...";
                setTimeout(() => window.location.href = '/login', 1500);

            } catch(err) {
                msg.className = "text-red-700 bg-red-100 p-3 rounded-lg block";
                msg.textContent = err.message;
                btn.disabled = false;
                btn.textContent = "Daftar Akun";
            }
        });
    </script>
</body>
</html>'''

INDEX_HTML = '''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignSpeak - Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
    <style>
        @keyframes pulse-ring {0%{transform:scale(0.8);opacity:0.5}100%{transform:scale(1.3);opacity:0}}
        .animate-pulse-ring::before{content:'';position:absolute;left:0;top:0;right:0;bottom:0;border-radius:50%;border:4px solid #ef4444;animation:pulse-ring 1.5s infinite}
    </style>
</head>
<body class="bg-gray-900 min-h-screen text-white">
    <script>
        // Auth Check
        if(!localStorage.getItem('signtara_token')){
            window.location.href='/login';
        }
    </script>

    <div class="container mx-auto px-4 py-8">
        <div class="flex justify-between items-center mb-8 bg-gray-800 p-4 rounded-2xl">
            <div class="flex items-center gap-4">
                <div class="text-left">
                    <h1 class="text-4xl font-bold">Signtara</h1>
                    <p class="text-gray-400 text-sm">Dashboard BISINDO Translator</p>
                </div>
            </div>

            <div class="flex gap-4">
                <div class="flex bg-gray-900 p-1.5 rounded-xl">
                    <button id="btnStatic" class="px-6 py-2.5 rounded-lg bg-indigo-600 font-bold text-sm">🅰️ ABJAD</button>
                    <button id="btnDynamic" class="px-6 py-2.5 rounded-lg text-gray-400 font-bold text-sm">👋 GERAKAN</button>
                </div>
                <a href="/premium" class="px-4 py-2 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-lg text-sm font-bold hover:shadow-lg transition">⭐ Premium</a>
                <button onclick="logout()" class="px-4 py-2 bg-red-600 rounded-lg text-sm font-bold hover:bg-red-700">Logout</button>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2">
                <div class="bg-gray-800 rounded-2xl p-4">
                    <div id="statusOverlay" class="absolute inset-0 bg-black/90 flex items-center justify-center rounded-2xl z-20">
                        <div class="text-center">
                            <div class="animate-spin rounded-full h-16 w-16 border-b-4 border-indigo-500 mx-auto mb-4"></div>
                            <p class="text-xl font-semibold">Memuat Kamera...</p>
                        </div>
                    </div>

                    <div class="relative bg-black rounded-xl overflow-hidden aspect-video">
                        <video id="video" class="hidden"></video>
                        <canvas id="output" class="w-full h-full object-contain"></canvas>
                        
                        <div class="absolute top-4 right-4 bg-black/60 px-4 py-1.5 rounded-full">
                            <span id="modeBadge" class="text-indigo-400 font-bold text-xs">MODE: ABJAD</span>
                        </div>

                        <div id="handCount" class="absolute top-4 left-4 bg-emerald-600/90 px-3 py-1 rounded-full text-xs hidden">
                            <span>🖐️</span> <span id="handCountText">0</span>
                        </div>

                        <div class="absolute bottom-4 right-4 bg-black/60 px-3 py-1 rounded-lg text-xs">
                            FPS: <span id="fpsCounter">0</span>
                        </div>

                        <div class="absolute bottom-4 left-4 bg-black border border-yellow-500/50 rounded-lg overflow-hidden hidden" id="debugContainer">
                            <canvas id="debugCrop" width="100" height="100"></canvas>
                            <p class="text-[10px] text-yellow-500 text-center bg-black/80 py-0.5">Input Model</p>
                        </div>

                        <div id="recordOverlay" class="absolute inset-0 flex items-center justify-center bg-black/40 hidden z-10">
                            <button id="btnRecord" class="w-24 h-24 rounded-full bg-red-600 flex flex-col items-center justify-center animate-pulse-ring">
                                <span class="text-3xl">🎥</span>
                                <span class="text-[10px] font-bold">REKAM</span>
                            </button>
                        </div>

                        <div id="recordingStatus" class="absolute bottom-8 left-1/2 transform -translate-x-1/2 bg-red-600/90 px-6 py-2 rounded-full hidden">
                            <span class="font-bold text-sm">⏺ Merekam...</span>
                        </div>
                    </div>

                    <div class="mt-3 flex justify-between px-2">
                        <p id="status" class="text-gray-400 text-xs">Status: Siap</p>
                        <div class="flex gap-3 text-xs">
                            <span id="stCnn" class="text-gray-500">CNN: ●</span>
                            <span id="stLstm" class="text-gray-500">LSTM: ●</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="space-y-6">
                <div class="bg-gray-800 rounded-2xl p-6 text-center">
                    <p class="text-gray-500 text-xs mb-3">HASIL TERDETEKSI</p>
                    <p id="predictResult" class="text-6xl font-black mb-2">-</p>
                    <p id="predictConfidence" class="text-indigo-400 text-sm">Menunggu...</p>
                    <div id="handDetectionStatus" class="mt-4 text-gray-500 text-xs">⏳ Menunggu tangan...</div>
                </div>

                <div class="bg-gray-800 rounded-2xl p-5">
                    <h3 class="font-bold mb-3 text-sm">📝 Susun Kalimat</h3>
                    <div class="bg-gray-900/50 p-4 rounded-xl mb-3 min-h-[60px]">
                        <p id="wordOutput" class="text-2xl font-bold">---</p>
                    </div>
                    <div class="grid grid-cols-2 gap-3">
                        <button id="addWordBtn" class="bg-emerald-600 py-2 rounded-lg font-bold text-sm">➕ Tambah</button>
                        <button id="clearWordBtn" class="bg-rose-600 py-2 rounded-lg font-bold text-sm">🗑️ Hapus</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin;
        const videoElement = document.getElementById("video");
        const canvasElement = document.getElementById("output");
        const context = canvasElement.getContext("2d");
        const debugCrop = document.getElementById("debugCrop");
        const debugCropCtx = debugCrop.getContext("2d");
        
        let CURRENT_MODE = "STATIC";
        let isRecording = false;
        let handsDetected = false;
        let lastHandLandmarks = null;
        let cnnIntervalId = null;
        let currentWord = "";
        let lastPredictedLabel = "";
        let lastPredictedConfidence = 0;
        let frameCount = 0;
        let lastFpsUpdate = Date.now();
        let CNN_PREDICT_INTERVAL = 800;
        let MIN_CONFIDENCE_DISPLAY = 0.30;
        let MIN_CONFIDENCE_FOR_ADD = 0.70;
        let REQUIRE_HAND_FOR_PREDICTION = true;

        const VIDEO_WIDTH = 640;
        const VIDEO_HEIGHT = 480;
        canvasElement.width = VIDEO_WIDTH;
        canvasElement.height = VIDEO_HEIGHT;

        function logout(){
            localStorage.removeItem('signtara_token');
            window.location.href='/';
        }

        function setMode(mode){
            CURRENT_MODE = mode;
            if(mode === "STATIC"){
                document.getElementById('btnStatic').className = "px-6 py-2.5 rounded-lg bg-indigo-600 font-bold text-sm";
                document.getElementById('btnDynamic').className = "px-6 py-2.5 rounded-lg text-gray-400 font-bold text-sm";
                document.getElementById('modeBadge').textContent = "MODE: ABJAD";
                document.getElementById('recordOverlay').classList.add("hidden");
                document.getElementById('predictResult').textContent = "-";
                document.getElementById('predictConfidence').textContent = "Menunggu...";
                startCnnLoop();
            }else{
                document.getElementById('btnDynamic').className = "px-6 py-2.5 rounded-lg bg-rose-600 font-bold text-sm";
                document.getElementById('btnStatic').className = "px-6 py-2.5 rounded-lg text-gray-400 font-bold text-sm";
                document.getElementById('modeBadge').textContent = "MODE: GERAKAN";
                document.getElementById('recordOverlay').classList.remove("hidden");
                document.getElementById('predictResult').textContent = "SIAP";
                document.getElementById('predictConfidence').textContent = "Tekan Rekam";
                stopCnnLoop();
            }
        }

        document.getElementById('btnStatic').onclick = () => setMode("STATIC");
        document.getElementById('btnDynamic').onclick = () => setMode("DYNAMIC");

        function updateFPS(){
            frameCount++;
            const now = Date.now();
            if(now - lastFpsUpdate >= 1000){
                document.getElementById('fpsCounter').textContent = frameCount;
                frameCount = 0;
                lastFpsUpdate = now;
            }
        }

        function getHandBoundingBox(landmarks) {
            let minX = 1, minY = 1, maxX = 0, maxY = 0;
            for (const lm of landmarks) {
                if (lm.x < minX) minX = lm.x;
                if (lm.y < minY) minY = lm.y;
                if (lm.x > maxX) maxX = lm.x;
                if (lm.y > maxY) maxY = lm.y;
            }
            const width = maxX - minX;
            const height = maxY - minY;
            const padding = 0.4;
            const size = Math.max(width, height) * (1 + padding);
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            let x1 = Math.max(0, centerX - size / 2);
            let y1 = Math.max(0, centerY - size / 2);
            return { x: x1, y: y1, size: Math.min(size, 1-x1, 1-y1) };
        }

        const hands = new Hands({locateFile:(file)=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
        hands.setOptions({maxNumHands:2,modelComplexity:1,minDetectionConfidence:0.6,minTrackingConfidence:0.5});

        hands.onResults((results)=>{
            context.save();
            context.clearRect(0,0,canvasElement.width,canvasElement.height);
            context.drawImage(results.image,0,0,canvasElement.width,canvasElement.height);

            if(results.multiHandLandmarks && results.multiHandLandmarks.length>0){
                handsDetected = true;
                lastHandLandmarks = results.multiHandLandmarks;
                document.getElementById('handCount').classList.remove("hidden");
                document.getElementById('handCountText').textContent = results.multiHandLandmarks.length;
                document.getElementById('handDetectionStatus').innerHTML = `✅ Terdeteksi`;
                
                results.multiHandLandmarks.forEach((landmarks)=>{
                    drawConnectors(context,landmarks,HAND_CONNECTIONS,{color:"#00FF00",lineWidth:2});
                    drawLandmarks(context,landmarks,{color:"#FF0000",lineWidth:1,radius:2});
                });
            }else{
                handsDetected = false;
                lastHandLandmarks = null;
                document.getElementById('handCount').classList.add("hidden");
                document.getElementById('handDetectionStatus').innerHTML = "⏳ Menunggu tangan...";
            }
            context.restore();
            updateFPS();
        });

        const cam = new Camera(videoElement,{
            onFrame:async()=>await hands.send({image:videoElement}),
            width:VIDEO_WIDTH,
            height:VIDEO_HEIGHT
        });

        cam.start().then(()=>{
            document.getElementById('statusOverlay').style.display="none";
            checkBackend();
        });

        function startCnnLoop(){
            if(cnnIntervalId) clearInterval(cnnIntervalId);
            cnnIntervalId = setInterval(predictCnn,CNN_PREDICT_INTERVAL);
        }

        function stopCnnLoop(){
            if(cnnIntervalId) clearInterval(cnnIntervalId);
            cnnIntervalId = null;
        }

        async function predictCnn(){
            if(CURRENT_MODE !== "STATIC" || isRecording) return;
            if(REQUIRE_HAND_FOR_PREDICTION && !handsDetected) return;

            try{
                const tempCanvas = document.createElement("canvas");
                const tCtx = tempCanvas.getContext("2d");
                tempCanvas.width = 224;
                tempCanvas.height = 224;
                
                // Smart Crop
                if(handsDetected && lastHandLandmarks){
                    const bbox = getHandBoundingBox(lastHandLandmarks[0]);
                    tCtx.drawImage(canvasElement, 
                        bbox.x * VIDEO_WIDTH, bbox.y * VIDEO_HEIGHT, bbox.size * VIDEO_WIDTH, bbox.size * VIDEO_HEIGHT,
                        0, 0, 224, 224);
                }else{
                    const size = Math.min(VIDEO_WIDTH,VIDEO_HEIGHT);
                    tCtx.drawImage(canvasElement,(VIDEO_WIDTH-size)/2,(VIDEO_HEIGHT-size)/2,size,size,0,0,224,224);
                }

                // Debug Crop Display
                if(document.getElementById('debugCropToggle').checked){
                    debugCropCtx.drawImage(tempCanvas, 0, 0, 100, 100);
                }

                const base64 = tempCanvas.toDataURL("image/jpeg",0.8).split(",")[1];
                
                const res = await fetch(`${API_BASE}/predict`,{
                    method:"POST",
                    headers:{"Content-Type":"application/json"},
                    body:JSON.stringify({image:base64})
                });

                const data = await res.json();
                if(data.label && data.confidence >= MIN_CONFIDENCE_DISPLAY){
                    document.getElementById('predictResult').textContent = data.label;
                    document.getElementById('predictConfidence').textContent = `${(data.confidence*100).toFixed(1)}% (CNN)`;
                    lastPredictedLabel = data.label;
                    lastPredictedConfidence = data.confidence;
                }else{
                    document.getElementById('predictResult').textContent = "?";
                    document.getElementById('predictConfidence').textContent = "Rendah";
                }
            }catch(e){console.error(e);}
        }

        document.getElementById('btnRecord').onclick = async()=>{
            if(isRecording) return;
            isRecording = true;
            
            document.getElementById('recordingStatus').classList.remove("hidden");
            document.getElementById('recordOverlay').classList.add("hidden");
            
            const stream = canvasElement.captureStream(30);
            const recorder = new MediaRecorder(stream,{mimeType:'video/webm'});
            const chunks = [];
            
            recorder.ondataavailable = e => chunks.push(e.data);
            
            recorder.onstop = async()=>{
                const blob = new Blob(chunks,{type:'video/webm'});
                const reader = new FileReader();
                reader.readAsDataURL(blob);
                
                reader.onloadend = async()=>{
                    const base64 = reader.result.split(",")[1];
                    document.getElementById('predictResult').textContent = "...";
                    document.getElementById('predictConfidence').textContent = "Menganalisis...";
                    
                    try{
                        const res = await fetch(`${API_BASE}/predict_lstm`,{
                            method:"POST",
                            headers:{"Content-Type":"application/json"},
                            body:JSON.stringify({video:base64})
                        });
                        
                        const data = await res.json();
                        document.getElementById('predictResult').textContent = data.label;
                        document.getElementById('predictConfidence').textContent = `${(data.confidence*100).toFixed(1)}% (LSTM)`;
                        lastPredictedLabel = data.label;
                        lastPredictedConfidence = data.confidence;
                    }catch(e){
                        document.getElementById('predictResult').textContent = "ERR";
                        document.getElementById('predictConfidence').textContent = "Gagal";
                    }
                    
                    isRecording = false;
                    document.getElementById('recordingStatus').classList.add("hidden");
                    document.getElementById('recordOverlay').classList.remove("hidden");
                };
            };
            
            recorder.start();
            setTimeout(()=>recorder.stop(),2000);
        };

        // Settings Handlers
        document.getElementById('intervalSlider').oninput = (e) => {
            CNN_PREDICT_INTERVAL = parseInt(e.target.value);
            document.getElementById('intervalValue').textContent = CNN_PREDICT_INTERVAL + " ms";
            if (CURRENT_MODE === "STATIC") { stopCnnLoop(); startCnnLoop(); }
        };
        
        document.getElementById('confidenceSlider').oninput = (e) => {
            MIN_CONFIDENCE_DISPLAY = e.target.value / 100;
            document.getElementById('confidenceValue').textContent = e.target.value + "%";
        };
        
        document.getElementById('handRequiredToggle').onchange = (e) => {
            REQUIRE_HAND_FOR_PREDICTION = e.target.checked;
        };
        
        document.getElementById('debugCropToggle').onchange = (e) => {
            if(e.target.checked) document.getElementById('debugContainer').classList.remove("hidden");
            else document.getElementById('debugContainer').classList.add("hidden");
        };

        document.getElementById('addWordBtn').onclick = ()=>{
            if(!lastPredictedLabel || lastPredictedConfidence < MIN_CONFIDENCE_FOR_ADD){
                document.getElementById('status').textContent = "Status: Hasil belum valid / confidence rendah";
                return;
            }
            const spacer = CURRENT_MODE==="DYNAMIC"?" ":"";
            currentWord += spacer + lastPredictedLabel;
            document.getElementById('wordOutput').textContent = currentWord;
            document.getElementById('status').textContent = `Status: Ditambahkan - ${lastPredictedLabel}`;
        };

        document.getElementById('clearWordBtn').onclick = ()=>{
            currentWord = "";
            document.getElementById('wordOutput').textContent = "---";
            document.getElementById('status').textContent = "Status: Kata dihapus";
        };

        // Upload Image Prediction
        document.getElementById('predictImageBtn').onclick = () => {
            const file = document.getElementById('imageInput').files[0];
            if(!file){
                document.getElementById('status').textContent = "Status: Pilih gambar dulu!";
                return;
            }
            const reader = new FileReader();
            reader.onload = async (e) => {
                const b64 = e.target.result.split(",")[1];
                try {
                    const res = await fetch(`${API_BASE}/predict`, {
                        method:"POST", 
                        headers:{"Content-Type":"application/json"},
                        body: JSON.stringify({image:b64})
                    });
                    const data = await res.json();
                    if(data.label) {
                        document.getElementById('predictResult').textContent = data.label;
                        document.getElementById('predictConfidence').textContent = `${(data.confidence*100).toFixed(1)}% (Upload)`;
                        document.getElementById('status').textContent = `Status: Upload - ${data.label}`;
                        lastPredictedLabel = data.label;
                        lastPredictedConfidence = data.confidence;
                    }
                } catch(e) {
                    document.getElementById('status').textContent = "Status: Gagal prediksi gambar";
                }
            };
            reader.readAsDataURL(file);
        };

        function checkBackend(){
            setInterval(()=>{
                fetch(`${API_BASE}/health`)
                    .then(r=>r.json())
                    .then(d=>{
                        document.getElementById('stCnn').className = d.cnn_ready?"text-emerald-400":"text-red-400";
                        document.getElementById('stCnn').textContent = d.cnn_ready?"CNN: ●":"CNN: ○";
                        document.getElementById('stLstm').className = d.lstm_ready?"text-emerald-400":"text-red-400";
                        document.getElementById('stLstm').textContent = d.lstm_ready?"LSTM: ●":"LSTM: ○";
                    })
                    .catch(()=>{});
            },3000);
        }

        startCnnLoop();
    </script>
</body>
</html>'''

PREMIUM_HTML = '''<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Premium - Signtara</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .blob {position:absolute;width:500px;height:500px;border-radius:50%;filter:blur(100px);opacity:0.3;animation:float 20s infinite}
        .blob-1{background:linear-gradient(45deg,#4F46E5,#7C3AED);top:10%;left:10%}
        .blob-2{background:linear-gradient(45deg,#EC4899,#F59E0B);bottom:10%;right:10%;animation-delay:5s}
        @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-50px)}}
    </style>
</head>
<body class="bg-gray-50 min-h-screen text-gray-900">
    <script>
        if(!localStorage.getItem('signtara_token')){
            window.location.href='/login';
        }
    </script>

    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>

    <header class="p-6 text-center relative z-10">
        <a href="/index" class="absolute left-6 top-6 text-indigo-600 font-bold hover:underline">← Kembali ke Dashboard</a>
        <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 mb-2">
            Signtara Premium
        </h1>
        <p class="text-gray-500">Buka potensi penuh komunikasi tanpa batas</p>
    </header>

    <div class="max-w-5xl mx-auto px-6 py-10 grid md:grid-cols-3 gap-8 relative z-10">
        
        <div class="bg-white/60 backdrop-blur-md rounded-2xl p-8 flex flex-col hover:scale-105 transition shadow-xl">
            <h3 class="text-xl font-bold text-gray-800 mb-2">Basic</h3>
            <div class="text-4xl font-bold text-gray-900 mb-6">Gratis</div>
            <ul class="space-y-4 mb-8 flex-1 text-gray-600 text-sm">
                <li>✅ Deteksi Alfabet Dasar</li>
                <li>✅ Akses Web App</li>
                <li>❌ Tanpa Mode Offline</li>
                <li>❌ Iklan Tampil</li>
            </ul>
            <button class="w-full py-3 rounded-xl border-2 border-indigo-600 text-indigo-600 font-bold hover:bg-indigo-50 transition">Paket Saat Ini</button>
        </div>

        <div class="bg-white/80 backdrop-blur-md rounded-2xl p-8 flex flex-col transform scale-105 ring-4 ring-indigo-200 relative shadow-2xl">
            <div class="absolute top-0 right-0 bg-indigo-600 text-white text-xs font-bold px-3 py-1 rounded-bl-xl rounded-tr-xl">POPULER</div>
            <h3 class="text-xl font-bold text-indigo-600 mb-2">Pro</h3>
            <div class="text-4xl font-bold text-gray-900 mb-1">Rp 49rb<span class="text-lg text-gray-400 font-normal">/bulan</span></div>
            <p class="text-xs text-gray-400 mb-6">Untuk pelajar & individu</p>
            <ul class="space-y-4 mb-8 flex-1 text-gray-600 text-sm">
                <li>✅ <b>Semua Fitur Basic</b></li>
                <li>✅ Deteksi Kata & Kalimat</li>
                <li>✅ Tanpa Iklan</li>
                <li>✅ Prioritas Server (Lebih Cepat)</li>
            </ul>
            <button class="w-full py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold shadow-lg hover:shadow-xl transition transform hover:-translate-y-1">Pilih Pro</button>
        </div>

        <div class="bg-white/60 backdrop-blur-md rounded-2xl p-8 flex flex-col hover:scale-105 transition shadow-xl">
            <h3 class="text-xl font-bold text-gray-800 mb-2">Sekolah / SLB</h3>
            <div class="text-4xl font-bold text-gray-900 mb-6">Hubungi Kami</div>
            <ul class="space-y-4 mb-8 flex-1 text-gray-600 text-sm">
                <li>✅ <b>Lisensi Institusi</b></li>
                <li>✅ API Akses Khusus</li>
                <li>✅ Integrasi Sistem Sekolah</li>
                <li>✅ Support 24/7</li>
            </ul>
            <button class="w-full py-3 rounded-xl bg-gray-900 text-white font-bold hover:bg-gray-800 transition">Kontak Sales</button>
        </div>
    </div>

    <footer class="py-8 text-center text-gray-500 relative z-10">
        © 2024 Signtara Project. Demo Version.
    </footer>
</body>
</html>
'''
=======
    return jsonify({
        "status": "healthy",
        "cnn_ready": bool(CNN_MODEL is not None),
        "cnn_classes": cnn_classes,
        "uptime_seconds": float(time.time() - _model_load_time) if _model_load_time else 0,
        "frontend_dir": str(FRONTEND_DIR),
        "frontend_exists": FRONTEND_DIR.exists()
    })
>>>>>>> 9c67073f5b774a3eb292f7627bc7bfd6abe18d66

# =========================
# 🚀 MAIN
# =========================
if __name__ == "__main__":
    ensure_dirs()
<<<<<<< HEAD
    log_event("🚀 Signtara Server Starting (All-in-One Demo)...")
    log_event(f"📍 Akses aplikasi di: http://127.0.0.1:5000")
    log_event(f"🔐 Demo Login - Username: demo, Password: demo123")
=======
    
    # Check frontend directory
    if FRONTEND_DIR.exists():
        log_event(f"✅ Frontend folder ditemukan: {FRONTEND_DIR}")
    else:
        log_event(f"⚠️ Frontend folder tidak ditemukan: {FRONTEND_DIR}")
        log_event("💡 Pastikan struktur folder:")
        log_event("   SIGNTARA/")
        log_event("   ├── backend/   ← app.py di sini")
        log_event("   └── frontend/  ← index.html di sini")
    
    log_event("🚀 SignSpeak Backend v5.0 (CNN-Only) starting...")
    
    # Load models on startup
>>>>>>> 9c67073f5b774a3eb292f7627bc7bfd6abe18d66
    load_all_models()
    
    try:
<<<<<<< HEAD
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        log_event("👋 Server dihentikan")
=======
        # Run Flask
        log_event("🌐 Server berjalan di:")
        log_event("   - http://127.0.0.1:5000")
        log_event("   - http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        log_event("⚠️ Keyboard interrupt received")
    finally:
        stop_event.set()
        poller.join(timeout=2)
        log_event("🛑 Server stopped.")
>>>>>>> 9c67073f5b774a3eb292f7627bc7bfd6abe18d66
