import os
import sys
import json
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# =========================
# 🔧 KONFIGURASI TARGET (MAX MODE)
# =========================
IMG_SIZE = (224, 224)
EPOCHS = 100            # Kita gas sampai 100 putaran
BATCH_SIZE = 32
ALPHA = 1.0
TARGET_ACCURACY = 0.995 # <--- UBAH INI (Target jadi 99.5%)

# =========================
# 🛑 CUSTOM CALLBACK
# =========================
class StopOnHighAccuracy(keras.callbacks.Callback):
    def __init__(self, target_acc):
        super(StopOnHighAccuracy, self).__init__()
        self.target_acc = target_acc

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        if val_acc is not None and val_acc >= self.target_acc:
            print(f"\n\n🎯 TARGET SULTAN TERCAPAI! Akurasi ({val_acc*100:.2f}%) >= {self.target_acc*100}%")
            print("🛑 Menghentikan training karena sudah sangat akurat...")
            self.model.stop_training = True

def main():
    print("="*60)
    print(f"🚀 TRAINING MAX PERFORMANCE MODE (Target: {TARGET_ACCURACY*100}%)")
    print("="*60)
    
    CURRENT_DIR = Path(__file__).resolve().parent
    MODELS_DIR = CURRENT_DIR / "models"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load Data
    print("📦 Memuat data .npy...")
    try:
        X = np.load("X_data.npy")
        y = np.load("y_data.npy")
        classes = np.load("class_map.npy", allow_pickle=True)
        print(f"✅ Data Loaded: {len(X)} sampel")
    except FileNotFoundError:
        print("❌ File .npy tidak ditemukan! Jalankan 'convert_to_npy.py' dulu.")
        sys.exit(1)

    # 2. Normalisasi
    X = X.astype('float32') / 255.0

    # 3. Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Arsitektur Model
    print("\n🏗️ Membangun Model...")
    
    # Augmentasi
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1), 
        layers.RandomZoom(0.1),     
    ])

    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=ALPHA
    )
    base.trainable = False 

    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        aug,
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Callbacks (Penyelamat)
    my_callbacks = [
        # PENTING: Ini akan menyimpan model TERBAIK saja.
        # Jadi kalau di epoch 100 hasilnya jelek (overfit), dia akan ambil yg epoch 80 (misalnya).
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "bisindo_cnn_best.keras"), 
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        # Turunkan learning rate kalau stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=5, 
            verbose=1,
            min_lr=1e-7
        ),
        # Stop kalau sudah 99.5%
        StopOnHighAccuracy(TARGET_ACCURACY)
    ]

    # 6. Training Phase 1 (Head Only)
    print(f"\n🔥 Mulai Training Phase 1 (Pemanasan)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20, 
        batch_size=BATCH_SIZE,
        callbacks=my_callbacks
    )

    if model.stop_training:
        print("✅ Target tercapai di Phase 1.")
    else:
        print("\n🔓 Phase 2: Fine Tuning (Mengejar Akurasi Tinggi)...")
        base.trainable = True
        # Fine tune layer 100 ke atas
        for layer in base.layers[:100]: layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), # LR Sangat Kecil biar detail
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            initial_epoch=history.epoch[-1] + 1,
            batch_size=BATCH_SIZE,
            callbacks=my_callbacks
        )

    # Simpan Akhir (Ini model terakhir, belum tentu terbaik)
    # Model terbaik ada di 'bisindo_cnn_best.keras'
    model.save(MODELS_DIR / "bisindo_cnn_final_epoch.keras")
    
    with open(MODELS_DIR / "class_names.json", "w") as f:
        json.dump(classes.tolist(), f)
        
    print("\n🎉 SELESAI! Gunakan file 'bisindo_cnn_best.keras' untuk hasil terbaik.")

if __name__ == "__main__":
    main()