import cv2
import os
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =========================
# 🔧 KONFIGURASI
# =========================
INPUT_DIR = Path("dataset_static")           # Folder dataset asli (Full Body)
OUTPUT_DIR = Path("dataset_cropped")  # Folder output (Hasil Crop)
IMG_SIZE = 224                        # Resolusi target (Wajib 224 untuk MobileNetV2)
MARGIN = 80                           # Margin diperbesar agar tidak terlalu mepet

# Init MediaPipemar
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,                  # ✅ PENTING: Deteksi maksimal 2 tangan
    min_detection_confidence=0.5
)

def get_unified_bbox(multi_hand_landmarks, h, w):
    """
    Menghitung bounding box yang mencakup SEMUA tangan yang terdeteksi.
    Otomatis menyesuaikan apakah itu 1 tangan atau 2 tangan.
    """
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    # Loop semua tangan yang terdeteksi (bisa 1 atau 2 tangan)
    for hand_landmarks in multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
    # Hitung lebar dan tinggi kotak
    box_w = x_max - x_min
    box_h = y_max - y_min
    
    # Jadikan kotak persegi (Square) agar proporsional saat di-resize
    max_side = max(box_w, box_h) + (MARGIN * 2)
    
    center_x = x_min + box_w // 2
    center_y = y_min + box_h // 2
    
    # Pastikan koordinat tidak keluar dari gambar
    x1 = max(0, center_x - max_side // 2)
    y1 = max(0, center_y - max_side // 2)
    x2 = min(w, center_x + max_side // 2)
    y2 = min(h, center_y + max_side // 2)
    
    return int(x1), int(y1), int(x2), int(y2)

def process_image(img_path, output_folder, file_prefix):
    """Memproses satu file gambar"""
    img = cv2.imread(str(img_path))
    if img is None: return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        h, w, _ = img.shape
        
        # Ambil koordinat kotak gabungan (Unified Bbox)
        x1, y1, x2, y2 = get_unified_bbox(results.multi_hand_landmarks, h, w)
        
        # Crop gambar
        crop = img[y1:y2, x1:x2]
        
        # Resize dan Simpan
        try:
            if crop.size == 0: return
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            save_path = output_folder / f"{file_prefix}_crop.jpg"
            cv2.imwrite(str(save_path), crop)
        except Exception:
            pass 

def process_video(video_path, output_folder, file_prefix):
    """Memproses file video (ambil beberapa frame)"""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Ambil setiap frame ke-5 (Sampling)
        if frame_count % 5 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                
                # Ambil koordinat kotak gabungan
                x1, y1, x2, y2 = get_unified_bbox(results.multi_hand_landmarks, h, w)
                
                crop = frame[y1:y2, x1:x2]
                
                try:
                    if crop.size == 0: continue
                    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                    save_path = output_folder / f"{file_prefix}_f{frame_count}.jpg"
                    cv2.imwrite(str(save_path), crop)
                except:
                    pass
        frame_count += 1
    cap.release()

def main():
    print("="*60)
    print("🚀 PRE-PROCESSING DATASET (DUAL HAND SUPPORT)")
    print("="*60)
    
    if not INPUT_DIR.exists():
        print(f"❌ Folder dataset asli tidak ditemukan: {INPUT_DIR}")
        print("   Pastikan nama folder dataset Anda adalah 'dataset'")
        return

    # Hapus folder output lama jika perlu (opsional)
    # import shutil
    # if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)

    classes = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    print(f"📂 Ditemukan {len(classes)} kelas untuk diproses.")
    
    for cls_dir in tqdm(classes, desc="Processing Classes"):
        output_cls_dir = OUTPUT_DIR / cls_dir.name
        output_cls_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(cls_dir.glob("*"))
        for f in files:
            ext = f.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                process_image(f, output_cls_dir, f.stem)
            elif ext in ['.mp4', '.avi', '.mov']:
                process_video(f, output_cls_dir, f.stem)

    print("\n✅ Selesai! Folder 'dataset_cropped' siap digunakan.")
    print("💡 Fitur: Mendukung 1 tangan maupun 2 tangan sekaligus.")
    print("👉 Langkah selanjutnya: Jalankan 'python train.py'")

main()