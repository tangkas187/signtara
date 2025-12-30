import os
# Paksa pakai Keras Baru untuk baca file aslinya
os.environ["TF_USE_LEGACY_KERAS"] = "0" 

import keras
from pathlib import Path

# Path File
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OLD_MODEL = MODELS_DIR / "bisindo_lstm_v2.keras"
NEW_MODEL = MODELS_DIR / "bisindo_lstm_v2.h5" # Kita ubah jadi .h5

print(f"🔄 Mengonversi model...")

try:
    # 1. Load pakai Keras 3 (Baru)
    model = keras.models.load_model(str(OLD_MODEL))
    print("✅ Berhasil load model .keras")

    # 2. Save pakai format H5 (Legacy Compatible)
    model.save(str(NEW_MODEL))
    print(f"✅ Berhasil convert ke: {NEW_MODEL.name}")
    print("🚀 Sekarang update model_utils.py kamu!")

except Exception as e:
    print(f"❌ Gagal convert: {e}")