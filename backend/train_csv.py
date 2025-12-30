import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================
# 🛑 CONFIG: PAKSA MODE LEGACY (WAJIB!)
# ============================================
# Ini biar formatnya .h5 yang bisa dibaca bareng CNN Teachable Machine
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tf_keras as keras
from tf_keras import layers

# =========================
# 🔧 KONFIGURASI
# =========================
CSV_FILE = "dataset_keypoints.csv"
OUTPUT_MODEL = "models/bisindo_lstm_csv.h5"  # Format .h5
OUTPUT_LABELS = "models/class_names_lstm.json"

# Sesuaikan dengan prepo2.py kamu
MAX_FRAMES = 20
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
# Total fitur per frame = 2 * 21 * 3 = 126
INPUT_FEATURES = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK 

def load_data(csv_path):
    print("📦 Memuat dataset CSV...")
    df = pd.read_csv(csv_path)
    
    # Pisahkan Label dan Fitur
    X = df.drop(columns=['label']).values
    y_str = df['label'].values
    
    # Cek shape data
    print(f"📊 Total Data Mentah: {X.shape}")
    
    # Encode Label (String -> Angka)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    classes = encoder.classes_
    
    # RESHAPE KE FORMAT LSTM (Samples, TimeSteps, Features)
    # Awalnya X itu flat (baris panjang), kita lipat jadi (20 frame, 126 fitur)
    num_samples = X.shape[0]
    
    # Validasi apakah kolom pas
    expected_cols = MAX_FRAMES * INPUT_FEATURES
    if X.shape[1] != expected_cols:
        raise ValueError(f"❌ Jumlah kolom CSV ({X.shape[1]}) tidak sesuai dengan konfigurasi ({expected_cols}). Cek prepo2.py!")
        
    X_reshaped = X.reshape(num_samples, MAX_FRAMES, INPUT_FEATURES)
    
    return X_reshaped, y, classes

def build_model(num_classes):
    model = keras.Sequential()
    
    # Input Layer (20 Frame, 126 Koordinat)
    model.add(layers.InputLayer(input_shape=(MAX_FRAMES, INPUT_FEATURES)))
    
    # LSTM Layers
    # return_sequences=True artinya outputnya diteruskan ke LSTM berikutnya
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.2))
    
    # Dense Layers (Otak Klasifikasi)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def main():
    # Buat folder models kalau belum ada
    os.makedirs("models", exist_ok=True)
    
    # 1. Load Data
    try:
        X, y, classes = load_data(CSV_FILE)
        print(f"✅ Data Ready! Shape: {X.shape}")
        print(f"🏷️ Kelas ({len(classes)}): {classes}")
    except FileNotFoundError:
        print(f"❌ File '{CSV_FILE}' tidak ditemukan! Jalankan prepo2.py dulu.")
        return

    # Split Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Build Model
    model = build_model(len(classes))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # 3. Training
    print("\n🔥 Mulai Training LSTM (CSV)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,       # Bisa ditambah kalau belum akurat
        batch_size=16,
        callbacks=[
            # Save format .h5 (Legacy)
            keras.callbacks.ModelCheckpoint(OUTPUT_MODEL, save_best_only=True, save_format="h5"),
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    # 4. Simpan Label
    with open(OUTPUT_LABELS, "w") as f:
        json.dump(classes.tolist(), f)
        
    print(f"\n✅ Training Selesai!")
    print(f"📁 Model tersimpan: {OUTPUT_MODEL}")
    print(f"📝 Label tersimpan: {OUTPUT_LABELS}")
    print("👉 Jangan lupa update app.py karena cara input datanya berubah (Video -> Landmark CSV)")

if __name__ == "__main__":
    main()