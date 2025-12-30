import os
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================
# 🛑 CONFIG: PAKSA MODE LEGACY (PENTING!)
# ============================================
# Ini bikin trainingnya pake 'otak' yang sama kayak Teachable Machine
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Import tf_keras (Pengganti keras biasa)
import tf_keras as keras
from tf_keras import layers, applications

# =========================
# 🔧 KONFIGURASI
# =========================
IMG_SIZE = 128
MAX_FRAMES = 20
BATCH_SIZE = 8 

def build_model(num_classes):
    # --- 1. Augmentasi Data ---
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # --- 2. CNN Base (MobileNetV2) ---
    # include_top=False artinya kita buang kepala klasifikasinya
    base_cnn = applications.MobileNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Bekukan base_cnn
    base_cnn.trainable = False 

    # --- 3. Rakit Model ---
    model = keras.Sequential()
    
    # Input Video
    model.add(layers.InputLayer(input_shape=(MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3)))
    
    # Augmentasi (TimeDistributed)
    model.add(layers.TimeDistributed(data_augmentation))
    
    # CNN (TimeDistributed)
    model.add(layers.TimeDistributed(base_cnn))
    model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))
    
    # LSTM
    model.add(layers.LSTM(64, return_sequences=False, dropout=0.5))
    
    # Classifier
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model, base_cnn

def main():
    CURRENT_DIR = Path(__file__).resolve().parent
    MODELS_DIR = CURRENT_DIR / "models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load Data
    try:
        print("📦 Memuat data...")
        # Pastikan file .npy ada di folder yang sama
        X = np.load("X_video.npy")
        y = np.load("y_video.npy")
        classes = np.load("class_map_lstm.npy", allow_pickle=True)
    except FileNotFoundError:
        print("❌ Data X_video.npy/y_video.npy tidak ditemukan!")
        sys.exit(1)

    # Normalisasi Data
    X = X.astype('float32') / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"✅ Data Ready. Train: {len(X_train)} | Val: {len(X_val)}")

    # 2. Build Model
    model, base_cnn = build_model(len(classes))
    model.summary()

    # ==========================================
    # 🚀 PHASE 1: WARM UP
    # ==========================================
    print("\n🔥 PHASE 1: Training LSTM & Dense (CNN Beku)...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10, 
        batch_size=BATCH_SIZE
    )

    # ==========================================
    # 🚀 PHASE 2: FINE TUNING
    # ==========================================
    print("\n🔓 Unfreezing CNN & Fine Tuning...")
    base_cnn.trainable = True
    
    # Sisakan 40 layer terakhir untuk dilatih
    for layer in base_cnn.layers[:-40]: 
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5), # LR Kecil
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🔥 PHASE 2: Training Full Model...")
    
    # PATH SAVE: PENTING! Kita ubah jadi .h5
    save_path = str(MODELS_DIR / "bisindo_lstm_v2.h5")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=BATCH_SIZE,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            # Save format .h5 otomatis kompatibel
            keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, save_format="h5")
        ]
    )
    
    # Simpan class names
    with open(MODELS_DIR / "class_names_lstm.json", "w") as f:
        json.dump(classes.tolist(), f)

    print("\n✅ Training Selesai.")
    print(f"📁 Model tersimpan di: {save_path}")

if __name__ == "__main__":
    main()