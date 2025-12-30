"""
model_utils_video.py
Helper functions untuk loading dan prediksi model VIDEO dengan motion features
"""

import json
import numpy as np
import cv2
from pathlib import Path
from tensorflow import keras

# =========================
# 📁 Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# =========================
# 🎥 VIDEO FEATURE EXTRACTION
# =========================
def extract_video_features(video_path, target_size=(128, 128), num_frames=30):
    """
    Extract frames and motion features from video
    Returns: (frames, motion_features) or (None, None) if failed
    """
    try:
        frames = []
        motion_features = []
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"⚠️ Tidak bisa membuka video: {video_path}")
            return None, None
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            print("⚠️ Video kosong")
            return None, None
        
        # Sample frames evenly
        indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
        
        prev_gray = None
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize and convert
            frame_resized = cv2.resize(frame, target_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Calculate optical flow (motion)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                # Motion magnitude
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_features.append(np.mean(mag))
            else:
                motion_features.append(0.0)
            
            prev_gray = gray
        
        cap.release()
        
        # Pad if not enough frames
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))
            motion_features.append(0.0)
        
        # Convert to numpy and normalize
        frames = np.array(frames[:num_frames], dtype=np.float32) / 255.0
        motion = np.array(motion_features[:num_frames], dtype=np.float32)
        
        return frames, motion
        
    except Exception as e:
        print(f"❌ Error extracting video features: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =========================
# 🧠 Load Video Model
# =========================
def load_video_model():
    """
    Load VIDEO model dan class names
    Returns: (model, class_names) atau (None, None) jika gagal
    """
    try:
        model_path = MODELS_DIR / "bisindo_video.keras"
        classes_path = MODELS_DIR / "class_names.json"
        
        # Check if files exist
        if not model_path.exists():
            print(f"⚠️ Model VIDEO tidak ditemukan: {model_path}")
            print("💡 Jalankan train_optimized.py terlebih dahulu!")
            return None, None
        
        if not classes_path.exists():
            print(f"⚠️ Class names tidak ditemukan: {classes_path}")
            return None, None
        
        # Load model
        print(f"🔥 Loading VIDEO model dari: {model_path}")
        model = keras.models.load_model(model_path)
        
        # Load class names
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        
        print(f"✅ VIDEO model loaded: {len(class_names)} classes")
        print(f"ℹ️ Model menggunakan OPTICAL FLOW untuk deteksi gerakan")
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading VIDEO model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =========================
# 🔮 Video Prediction
# =========================
def predict_video_model(model, class_names, frames, motion):
    """
    Prediksi menggunakan VIDEO model dengan motion features
    
    Args:
        model: Keras model (expects 2 inputs: frames and motion)
        class_names: List of class names
        frames: Video frames (num_frames, height, width, 3)
        motion: Motion features (num_frames,)
    
    Returns:
        (label, confidence)
    """
    if model is None or class_names is None:
        raise ValueError("Model atau class names belum dimuat")
    
    # Prepare inputs
    frames_input = np.expand_dims(frames, axis=0)  # (1, seq_len, h, w, 3)
    motion_input = np.expand_dims(motion, axis=0)   # (1, seq_len)
    
    # Predict
    predictions = model.predict([frames_input, motion_input], verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_idx])
    
    # Get label
    if pred_idx < len(class_names):
        label = class_names[pred_idx]
    else:
        label = f"class_{pred_idx}"
    
    return label, confidence

# =========================
# 🧪 Test Functions
# =========================
def test_model():
    """Test apakah model bisa dimuat"""
    print("="*60)
    print("🧪 Testing VIDEO Model Loading...")
    print("="*60)
    
    model, class_names = load_video_model()
    if model is not None:
        print(f"✅ VIDEO Model: OK ({len(class_names)} classes)")
        print(f"📊 Model inputs: {[inp.name for inp in model.inputs]}")
        print(f"📊 Model outputs: {model.output.shape}")
        
        # Test with dummy data
        print("\n🧪 Testing with dummy data...")
        dummy_frames = np.random.rand(30, 128, 128, 3).astype(np.float32)
        dummy_motion = np.random.rand(30).astype(np.float32)
        
        try:
            label, conf = predict_video_model(model, class_names, dummy_frames, dummy_motion)
            print(f"✅ Dummy prediction: {label} ({conf:.4f})")
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
    else:
        print("❌ VIDEO Model: FAILED")
    
    print("="*60)

if __name__ == "__main__":
    test_model()