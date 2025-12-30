import numpy as np
import json
import os
from pathlib import Path

def main():
    print("🚑 MEMPERBAIKI FILE JSON YANG HILANG...")
    
    # 1. Cari file source (hasil prepo)
    source_file = "class_map_lstm.npy"
    
    if not os.path.exists(source_file):
        print(f"❌ Gawat! File '{source_file}' tidak ditemukan.")
        print("   Pastikan kamu menjalankan script ini di folder yang sama dengan file .npy")
        return

    # 2. Load data kelas
    classes = np.load(source_file, allow_pickle=True)
    print(f"✅ Ditemukan {len(classes)} kelas: {classes}")

    # 3. Tentukan folder tujuan (models_torch)
    dest_folder = Path("models_torch")
    dest_folder.mkdir(exist_ok=True) # Buat folder kalau belum ada
    
    dest_file = dest_folder / "class_names.json"

    # 4. Simpan ke JSON
    try:
        with open(dest_file, "w") as f:
            json.dump(classes.tolist(), f)
        print(f"🎉 SUKSES! File tersimpan di: {dest_file}")
        print("👉 Sekarang coba restart app.py kamu.")
    except Exception as e:
        print(f"❌ Gagal menyimpan: {e}")

if __name__ == "__main__":
    main()