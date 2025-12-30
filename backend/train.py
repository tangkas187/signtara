import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

# =========================
# 🔧 KONFIGURASI
# =========================
IMG_SIZE = (224, 224)   # Wajib 224x224 untuk MobileNetV2
EPOCHS = 25             # Jumlah putaran belajar
BATCH_SIZE = 32         # Jumlah gambar per batch
TARGET_SAMPLES = 150    # Target jumlah foto per kelas (Balancing)
ALPHA = 1.0             # 1.0 = Model Pintar (Standar), 0.35 = Model Cepat (Lite)

def clear_old_models(models_dir):
    """Membersihkan file model lama sebelum training baru"""
    if models_dir.exists():
        print(f"🧹 Membersihkan folder model lama: {models_dir}")
        count = 0
        for item in models_dir.iterdir():
            try:
                if item.is_file() and item.suffix in [".keras", ".h5", ".json"]:
                    item.unlink()
                    count += 1
            except Exception as e:
                print(f"  ⚠️ {item.name}: {e}")
        print(f"✅ {count} file dibersihkan.\n")

def load_dataset(base_dir, img_height, img_width):
    """
    Hanya memuat gambar (.jpg, .png) dari folder dataset.
    Tidak ada video.
    """
    X, y = [], []
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        print(f"❌ Folder dataset tidak ditemukan: {base_dir}")
        return np.array([]), np.array([])
    
    all_classes = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    print(f"📂 Ditemukan {len(all_classes)} kelas: {all_classes}\n")

    for cls in all_classes:
        cls_path = base_dir / cls
        files = list(cls_path.glob("*"))
        count = 0
        
        # Filter hanya file gambar
        valid_files = [f for f in files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

        for f in tqdm(valid_files, desc=f"Loading {cls:15s}", leave=False, ncols=80):
            try:
                # Baca gambar
                img = cv2.imread(str(f))
                if img is None: continue
                
                # Resize ke 224x224 dan ubah ke RGB
                img = cv2.cvtColor(cv2.resize(img, (img_width, img_height)), cv2.COLOR_BGR2RGB)
                
                X.append(img)
                y.append(cls)
                count += 1
            except Exception:
                continue
        
        print(f"  ✅ {cls:15s}: {count:4d} foto")

    if not X:
        print("❌ Tidak ada data foto yang berhasil dimuat!")
        return np.array([]), np.array([])
    
    # Konversi ke numpy array & Normalisasi (0-1)
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)
    
    print(f"\n✅ Total data dimuat: {len(X)} foto dari {len(set(y))} kelas")
    return X, y

# =========================
# 🚀 MAIN TRAINING
# =========================
def main():
    print("="*70)
    print("🚀 TRAINING BISINDO CNN – FOTO ONLY (STATIC)")
    print("="*70)

    CURRENT_DIR = Path(__file__).resolve().parent
    DATASET_DIR = CURRENT_DIR / "dataset_cropped" # Pastikan nama folder ini benar
    MODELS_DIR = CURRENT_DIR / "models"
    
    # Persiapan Folder Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    clear_old_models(MODELS_DIR)

    # 1. Load Data (Hanya Foto)
    print(f"📦 Memuat dataset dari: {DATASET_DIR}")
    X, y = load_dataset(DATASET_DIR, *IMG_SIZE)
    
    if len(X) == 0:
        print("\n❌ Dataset kosong! Pastikan folder 'dataset_cropped' berisi foto.")
        sys.exit(1)

    # 2. Balancing (Penting agar tidak bias)
    unique_classes = sorted(set(y))
    label_map = {c: i for i, c in enumerate(unique_classes)}
    y_enc = np.array([label_map[label] for label in y])
    num_classes = len(unique_classes)

    print(f"\n⚖️ Balancing dataset (Target: {TARGET_SAMPLES} foto per kelas)...")
    X_bal, y_bal = [], []
    
    for cls in unique_classes:
        idxs = np.where(y == cls)[0]
        current_count = len(idxs)
        
        if current_count < TARGET_SAMPLES:
            # Jika kurang, duplikat foto secara acak (Oversampling)
            chosen = np.random.choice(idxs, TARGET_SAMPLES, replace=True)
        else:
            # Jika lebih, ambil sebagian secara acak (Undersampling)
            chosen = np.random.choice(idxs, TARGET_SAMPLES, replace=False)
        
        X_bal.extend(X[i] for i in chosen)
        y_bal.extend(y_enc[i] for i in chosen)

    X_bal = np.array(X_bal, dtype=np.float32)
    y_bal = np.array(y_bal, dtype=np.int32)
    
    # 3. Split Data (80% Latih, 20% Ujian)
    X_train, X_val, y_train, y_val = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )
    print(f"✅ Siap Training: {len(X_train)} data latih | {len(X_val)} data validasi")

    # 4. Bangun Model CNN (MobileNetV2)
    print("\n🏗️ Membangun Arsitektur Model...")
    
    # Augmentasi Data (Biar model kenal variasi posisi/cahaya)
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15), # Putar-putar sedikit
        layers.RandomZoom(0.1),      # Zoom in/out sedikit
        layers.RandomContrast(0.1),  # Ubah kontras
        layers.RandomBrightness(0.1),# Ubah kecerahan
    ], name="augmentation")

    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=ALPHA
    )
    base.trainable = False # Bekukan dulu

    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        aug,
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4), # Dropout agak besar biar ga overfitting
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Callbacks (Pengontrol Latihan)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "bisindo_cnn_best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        )
    ]

    # 6. Phase 1: Transfer Learning
    print(f"\n🚀 Phase 1: Belajar Pola Dasar ({EPOCHS} epochs)...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # 7. Phase 2: Fine-Tuning (Biar lebih jago detail)
    print(f"\n🔬 Phase 2: Fine-Tuning (Mempertajam Akurasi)...")
    base.trainable = True
    for layer in base.layers[:-30]: # Buka 30 layer terakhir
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Learning rate sangat kecil
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS + 10,
        initial_epoch=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    # 8. Simpan & Selesai
    print(f"\n💾 Menyimpan Hasil Akhir...")
    model.save(MODELS_DIR / "bisindo_cnn.keras")
    
    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(unique_classes, f, indent=2)
    
    print("\n🎉 TRAINING SELESAI!")
    print(f"✅ Model tersimpan di folder 'models/'")
    print(f"👉 Jalankan 'app.py' untuk mencoba.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
<<<<<<< HEAD
        print(f"❌ Error: {e}")
=======
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
>>>>>>> 9c67073f5b774a3eb292f7627bc7bfd6abe18d66
