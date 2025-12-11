from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback
from tensorflow import keras
import cv2
import base64
import json
import os
from pathlib import Path
import tempfile

# --- Tambahan untuk KNN ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# --- Direktori ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = BASE_DIR / "temp_videos"

KNN_MODEL_PATH = MODELS_DIR / "knn_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
LBLENC_PATH = MODELS_DIR / "label_encoder.joblib"
CNN_MODEL_PATH = MODELS_DIR / "bisindo_cnn.h5"
CLASS_FILE_PATH = MODELS_DIR / "class_names.json"
SAMPLES_FILE = DATA_DIR / "samples.jsonl"

def ensure_dirs():
    """Memastikan semua direktori yang dibutuhkan ada."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)

def load_knn_samples():
    """Memuat sampel data untuk model KNN dari file JSONL."""
    ensure_dirs()
    if not SAMPLES_FILE.exists():
        return np.empty((0, 63)), np.array([])
    
    X, y = [], []
    with open(SAMPLES_FILE, "r") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                # Pastikan jumlah fitur sesuai (63 untuk satu tangan, 126 untuk dua tangan)
                if "features" in obj and "label" in obj and len(obj["features"]) in [63, 126]:
                    X.append(obj["features"])
                    y.append(obj["label"])
            except (json.JSONDecodeError, KeyError):
                print(f"Melewatkan baris data yang tidak valid: {line.strip()}")
                continue
                
    if not X:
        return np.empty((0, 63)), np.array([])
        
    return np.array(X, dtype=float), np.array(y)

# --- Setup Flask ---
app = Flask(__name__)
CORS(app)

MODEL = None
CLASS_NAMES = None
IMG_HEIGHT = 64
IMG_WIDTH = 64
KNN_MODEL = None
SCALER = None
LABEL_ENCODER = None

# --- Load model saat startup ---
try:
    print(f"⌛ Memuat model CNN dari {CNN_MODEL_PATH}")
    if CNN_MODEL_PATH.exists():
        MODEL = keras.models.load_model(CNN_MODEL_PATH)
        print("✅ Model CNN berhasil dimuat.")
    else:
        print("⚠️ Model CNN tidak ditemukan. Jalankan train.py terlebih dahulu.")
        
    if CLASS_FILE_PATH.exists():
        with open(CLASS_FILE_PATH, "r") as f:
            CLASS_NAMES = json.load(f)
            print(f"✅ Class names berhasil dimuat: {CLASS_NAMES}")
    else:
        print("⚠️ class_names.json tidak ditemukan.")
        
    if KNN_MODEL_PATH.exists() and SCALER_PATH.exists() and LBLENC_PATH.exists():
        KNN_MODEL = joblib.load(KNN_MODEL_PATH)
        SCALER = joblib.load(SCALER_PATH)
        LABEL_ENCODER = joblib.load(LBLENC_PATH)
        print("✅ Model KNN berhasil dimuat.")
    else:
        print("ℹ️ Model KNN belum dilatih. Gunakan endpoint /knn/train untuk melatih.")
except Exception as e:
    print(f"❌ Gagal memuat model saat startup: {e}")
    traceback.print_exc()

# =================================================================
# 🧠 ENDPOINT CNN (Gambar & Video)
# =================================================================

@app.route("/")
def index():
    """Endpoint utama untuk mengecek status server."""
    return jsonify({
        "status": "ok",
        "model_cnn_loaded": MODEL is not None,
        "model_knn_loaded": KNN_MODEL is not None,
        "classes": CLASS_NAMES if CLASS_NAMES else [],
        "version": "2.0 - Dynamic Motion Support"
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Prediksi CNN dari gambar (base64 snapshot)."""
    try:
        if MODEL is None:
            return jsonify({"error": "Model CNN belum dimuat. Jalankan train.py terlebih dahulu."}), 503
        
        data = request.json.get("image")
        if not data:
            return jsonify({"error": "Gambar tidak ditemukan dalam request"}), 400

        image_data = base64.b64decode(data)
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Gagal men-decode data gambar"}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img_array = np.expand_dims(img.astype("float32") / 255.0, axis=0)

        preds = MODEL.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = CLASS_NAMES[idx] if CLASS_NAMES and idx < len(CLASS_NAMES) else f"class_{idx}"
        confidence = float(preds[idx])
        
        print(f"🔮 Prediksi Snapshot: {label} ({confidence*100:.1f}%)")
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_video", methods=["POST"])
def predict_video():
    """Prediksi CNN dari video (base64 .webm/.mp4)."""
    temp_path = None
    try:
        if MODEL is None:
            return jsonify({"error": "Model CNN belum dimuat"}), 503

        data = request.json.get("video")
        if not data:
            return jsonify({"error": "Video tidak ditemukan dalam request"}), 400

        print("📹 Menerima video untuk prediksi...")
        
        video_data = base64.b64decode(data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm', dir=str(VIDEOS_DIR)) as tmp:
            tmp.write(video_data)
            temp_path = tmp.name
        
        print(f"💾 Video disimpan sementara di: {temp_path} ({len(video_data)} bytes)")

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({"error": "Gagal membuka file video. Format mungkin tidak didukung."}), 400

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Info video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")

        max_frames_to_process = 30
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames_to_process, total_frames), dtype=int)
        
        frame_preds = []
        frames_processed = 0
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT))
            frame_array = np.expand_dims(frame_resized.astype("float32") / 255.0, axis=0)
            
            pred = MODEL.predict(frame_array, verbose=0)[0]
            frame_preds.append(pred)
            frames_processed += 1

        cap.release()
        
        if not frame_preds:
            return jsonify({"error": "Video kosong atau tidak dapat diproses"}), 400

        print(f"✅ Selesai memproses {frames_processed} frame")

        avg_pred = np.mean(frame_preds, axis=0)
        top_indices = np.argsort(avg_pred)[-3:][::-1]
        
        results = []
        for i in top_indices:
            label = CLASS_NAMES[i] if CLASS_NAMES and i < len(CLASS_NAMES) else f"class_{i}"
            conf = float(avg_pred[i])
            results.append({"label": label, "confidence": conf})
        
        best_result = results[0]
        print(f"🎯 Prediksi Video: {best_result['label']} ({best_result['confidence']*100:.1f}%)")
        # --- PERBAIKAN F-STRING DI SINI ---
        # Menggunakan kutip tunggal (') untuk f-string bagian dalam
        # dan kutip ganda (") untuk akses key dictionary.
        top_3_str = ', '.join([f'{r["label"]}({r["confidence"]*100:.0f}%)' for r in results])
        print(f"   Top 3: {top_3_str}")
        
        return jsonify({
            "label": best_result["label"],
            "confidence": best_result["confidence"],
            "top3": results,
            "frames_analyzed": frames_processed
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Selalu pastikan file temporary dihapus
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️ File temporary dihapus: {temp_path}")
            except OSError as e:
                print(f"Gagal menghapus file temporary {temp_path}: {e}")

# =================================================================
# 🧩 ENDPOINT KNN (Landmark)
# =================================================================

@app.route("/knn/add_sample", methods=["POST"])
def knn_add_sample():
    """Menambahkan sampel data landmark untuk training KNN."""
    try:
        data = request.json
        label = data.get("label")
        features = data.get("features")
        
        # Validasi yang lebih fleksibel untuk 1 atau 2 tangan
        if not label or not features or not isinstance(features, list) or len(features) not in [63, 126]:
            return jsonify({"error": "Data tidak valid. Butuh 'label' dan 'features' (list dengan 63 atau 126 elemen)."}), 400
        
        ensure_dirs()
        with open(SAMPLES_FILE, "a") as f:
            f.write(json.dumps({"label": label, "features": features}) + "\n")
            
        X, _ = load_knn_samples()
        print(f"✅ Sampel ditambahkan: {label} (Total sampel sekarang: {len(X)})")
        return jsonify({"message": "Sampel berhasil ditambahkan", "total_samples": len(X)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/knn/train", methods=["POST"])
def knn_train():
    """Melatih model KNN berdasarkan sampel yang ada."""
    global KNN_MODEL, SCALER, LABEL_ENCODER
    try:
        X, y = load_knn_samples()
        if len(X) < 5: # Butuh lebih banyak sampel untuk hasil yang baik
            return jsonify({"error": f"Sampel belum cukup (minimal 5, sekarang: {len(X)})"}), 400

        unique_classes = len(np.unique(y))
        if unique_classes < 2:
            return jsonify({"error": f"Minimal dibutuhkan 2 kelas berbeda untuk training (sekarang: {unique_classes})"}), 400

        SCALER = StandardScaler()
        X_scaled = SCALER.fit_transform(X)
        LABEL_ENCODER = LabelEncoder()
        y_encoded = LABEL_ENCODER.fit_transform(y)
        
        # Menentukan jumlah tetangga (k) secara dinamis
        n_neighbors = min(5, len(X) // unique_classes)
        if n_neighbors < 1: n_neighbors = 1

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        KNN_MODEL = KNeighborsClassifier(n_neighbors=n_neighbors)
        KNN_MODEL.fit(X_train, y_train)
        accuracy = KNN_MODEL.score(X_test, y_test)
        
        ensure_dirs()
        joblib.dump(KNN_MODEL, KNN_MODEL_PATH)
        joblib.dump(SCALER, SCALER_PATH)
        joblib.dump(LABEL_ENCODER, LBLENC_PATH)
        
        print(f"✅ Model KNN dilatih: {unique_classes} kelas, akurasi {accuracy*100:.1f}%")
        return jsonify({
            "message": "Model KNN berhasil dilatih dan disimpan", 
            "accuracy": accuracy,
            "classes_count": len(LABEL_ENCODER.classes_),
            "total_samples": len(X)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/knn/predict", methods=["POST"])
def knn_predict():
    """Memprediksi label dari data landmark menggunakan KNN."""
    try:
        if not all([KNN_MODEL, SCALER, LABEL_ENCODER]):
            return jsonify({"error": "Model KNN, Scaler, atau LabelEncoder belum dilatih/dimuat"}), 503
        
        features = request.json.get("features")
        if not features or len(features) not in [63, 126]:
            # Mengembalikan hasil default jika tidak ada fitur, agar frontend tidak error
            return jsonify({"label": "?", "confidence": 0.0}), 200
            
        X_scaled = SCALER.transform(np.array(features).reshape(1, -1))
        probs = KNN_MODEL.predict_proba(X_scaled)[0]
        idx = int(np.argmax(probs))
        
        label = LABEL_ENCODER.classes_[idx]
        confidence = float(probs[idx])
        
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        print(f"⚠️ Error pada prediksi KNN: {e}")
        # Mengembalikan default jika ada error
        return jsonify({"label": "?", "confidence": 0.0}), 200


@app.route("/health", methods=["GET"])
def health():
    """Endpoint untuk health check."""
    return jsonify({
        "status": "healthy",
        "cnn_ready": MODEL is not None,
        "knn_ready": KNN_MODEL is not None,
        "classes": CLASS_NAMES if CLASS_NAMES else [],
        "temp_dir_exists": VIDEOS_DIR.exists()
    })


if __name__ == "__main__":
    ensure_dirs()
    print("\n" + "="*60)
    print("🚀 SignSpeak Backend Server v2.0")
    print("="*60)
    print(f"📁 Direktori Model: {MODELS_DIR}")
    print(f"📁 Direktori Data: {DATA_DIR}")
    print(f"📁 Direktori Video Temp: {VIDEOS_DIR}")
    print(f"🧠 Model CNN: {'✅ Dimuat' if MODEL else '❌ Tidak Ditemukan'}")
    print(f"🧩 Model KNN: {'✅ Dimuat' if KNN_MODEL else '⚠️ Belum Dilatih'}")
    if CLASS_NAMES:
        print(f"📋 Kelas Terdeteksi ({len(CLASS_NAMES)}): {', '.join(CLASS_NAMES)}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)

