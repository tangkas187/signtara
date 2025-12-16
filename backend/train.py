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
# 🔧 CONFIG
# =========================
IMG_SIZE = (128, 128)
EPOCHS = 25
BATCH_SIZE = 32
MAX_FRAMES = 15
TARGET_SAMPLES = 120
ALPHA = 0.35  # MobileNetV2 width multiplier

# =========================
# 🧰 Utility
# =========================
def clear_old_models(models_dir):
    """Clear old model files from models directory"""
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

def extract_frames_from_video(video_path, target_size=(128, 128), max_frames=15):
    """Extract frames evenly from video"""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return frames
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return frames
    
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(cv2.resize(frame, target_size), cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return frames

def load_dataset(base_dir, img_height=128, img_width=128, max_frames=15):
    """Load dataset from directory with images and videos"""
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
        frames_collected = 0
        
        for f in tqdm(files, desc=f"Loading {cls:15s}", leave=False, ncols=80):
            ext = f.suffix.lower()
            
            # Process images
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                img = cv2.cvtColor(cv2.resize(img, (img_width, img_height)), cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(cls)
                frames_collected += 1
            
            # Process videos
            elif ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                frames = extract_frames_from_video(f, (img_width, img_height), max_frames)
                X.extend(frames)
                y.extend([cls] * len(frames))
                frames_collected += len(frames)
        
        print(f"  ✅ {cls:15s}: {frames_collected:4d} sampel")

    if not X:
        print("❌ Tidak ada data yang berhasil dimuat!")
        return np.array([]), np.array([])
    
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)
    
    print(f"\n✅ Total data dimuat: {len(X)} sampel dari {len(set(y))} kelas")
    return X, y

# =========================
# 🚀 MAIN TRAINING FUNCTION
# =========================
def main():
    print("="*70)
    print("🚀 TRAINING BISINDO CNN – OPTIMIZED VERSION")
    print("="*70)

    CURRENT_DIR = Path(__file__).resolve().parent
    DATASET_DIR = CURRENT_DIR / "dataset"
    MODELS_DIR = CURRENT_DIR / "models"
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    clear_old_models(MODELS_DIR)

    # Load dataset
    print(f"📦 Memuat dataset dari: {DATASET_DIR}")
    X, y = load_dataset(DATASET_DIR, *IMG_SIZE, max_frames=MAX_FRAMES)
    
    if len(X) == 0:
        print("\n❌ Dataset kosong atau tidak valid!")
        print("💡 Pastikan folder dataset/ berisi subfolder dengan gambar/video")
        sys.exit(1)

    # Display class distribution
    print("\n📊 Distribusi Kelas:")
    class_counts = Counter(y)
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:15s}: {count:4d} sampel")

    unique_classes = sorted(set(y))
    label_map = {c: i for i, c in enumerate(unique_classes)}
    y_enc = np.array([label_map[label] for label in y])
    num_classes = len(unique_classes)

    # Balance dataset
    print(f"\n⚖️ Balancing dataset (target: {TARGET_SAMPLES} per kelas)...")
    X_bal, y_bal = [], []
    
    for cls in unique_classes:
        idxs = np.where(y == cls)[0]
        if len(idxs) < TARGET_SAMPLES:
            # Oversample if not enough samples
            chosen = np.random.choice(idxs, TARGET_SAMPLES, replace=True)
        else:
            # Undersample if too many samples
            chosen = np.random.choice(idxs, TARGET_SAMPLES, replace=False)
        
        X_bal.extend(X[i] for i in chosen)
        y_bal.extend(y_enc[i] for i in chosen)

    X_bal = np.array(X_bal, dtype=np.float32)
    y_bal = np.array(y_bal, dtype=np.int32)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )
    print(f"✅ Train: {len(X_train)} | Val: {len(X_val)}")

    # Build model
    print("\n🏗️ Membangun model CNN...")
    
    # Data augmentation layers
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.1),
    ], name="augmentation")

    # Base model (MobileNetV2)
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        alpha=ALPHA
    )
    base.trainable = False

    # Complete model
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        aug,
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.35),
        layers.Dense(256, activation="relu", name="embedding"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax", name="predictions")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"✅ Model dibuat dengan {num_classes} kelas output")
    print(f"📊 Total parameters: {model.count_params():,}")

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "bisindo_cnn_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        )
    ]

    # Training phase 1: Transfer learning
    print(f"\n🚀 Phase 1: Training CNN (Transfer Learning) - {EPOCHS} epochs...")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Training phase 2: Fine-tuning
    print(f"\n🔬 Phase 2: Fine-tuning last 25 layers - {10} epochs...")
    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS + 10,
        initial_epoch=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save CNN model
    print(f"\n💾 Menyimpan model CNN...")
    model.save(MODELS_DIR / "bisindo_cnn.keras")
    
    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(unique_classes, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Model CNN disimpan di: {MODELS_DIR / 'bisindo_cnn.keras'}")

    # Final evaluation
    print("\n📊 EVALUASI AKHIR:")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  CNN Validation Accuracy : {val_acc*100:.2f}%")
    print(f"  CNN Validation Loss     : {val_loss:.4f}")

    # Save training summary
    summary = {
        "timestamp": str(np.datetime64('now')),
        "dataset": {
            "total_samples": len(X),
            "balanced_samples": len(X_bal),
            "num_classes": num_classes,
            "classes": unique_classes
        },
        "cnn": {
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "epochs": EPOCHS + 10
        }
    }
    
    with open(MODELS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n🎉 TRAINING SELESAI!")
    print("="*70)
    print(f"📁 Model tersimpan di: {MODELS_DIR}")
    print("\n💡 Langkah selanjutnya:")
    print("  1. Jalankan: python app.py")
    print("  2. Buka index.html di browser")
    print("  3. Mulai prediksi!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training dibatalkan oleh user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
