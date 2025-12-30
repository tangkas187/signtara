import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =========================
# 🔧 KONFIGURASI
# =========================
DATASET_DIR = Path("dataset_cropped")  # Folder hasil crop
IMG_SIZE = 224                         # Harus sama dengan training
OUTPUT_FILE_X = "X_data.npy"
OUTPUT_FILE_Y = "y_data.npy"
OUTPUT_CLASS_MAP = "class_map.npy"     # Menyimpan urutan kelas

def main():
    if not DATASET_DIR.exists():
        print("❌ Folder dataset_cropped tidak ditemukan!")
        return

    print(f"🚀 Sedang mengemas dataset ke format NPY (Cepat)...")
    
    classes = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
    class_map = {c: i for i, c in enumerate(classes)}
    
    X = []
    y = []
    
    print(f"📂 Ditemukan {len(classes)} kelas: {classes}")

    for cls in classes:
        cls_path = DATASET_DIR / cls
        files = list(cls_path.glob("*"))
        valid_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        for f in tqdm(valid_files, desc=f"Loading {cls}", leave=False):
            try:
                img = cv2.imread(str(f))
                if img is None: continue
                
                # Resize & Convert RGB
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                X.append(img)
                y.append(class_map[cls])
            except:
                continue

    # Convert ke Numpy Array
    print("\n📦 Sedang mengkonversi ke NumPy array (Makan RAM)...")
    X = np.array(X, dtype=np.uint8) # Pakai uint8 biar hemat RAM & Storage
    y = np.array(y, dtype=np.uint8)
    
    print(f"💾 Menyimpan ke file .npy...")
    np.save(OUTPUT_FILE_X, X)
    np.save(OUTPUT_FILE_Y, y)
    np.save(OUTPUT_CLASS_MAP, classes)
    
    print("\n✅ SELESAI! Data berhasil dipacking.")
    print(f"   X Shape: {X.shape}")
    print(f"   Y Shape: {y.shape}")
    print("👉 Sekarang jalankan 'train_fast.py'")

if __name__ == "__main__":
    main()