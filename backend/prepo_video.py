import cv2
import os
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# =========================
# 🔧 KONFIGURASI VIDEO
# =========================
INPUT_DIR = Path("dataset_dynamic")
OUTPUT_FILE_X = "X_video.npy"
OUTPUT_FILE_Y = "y_video.npy"
OUTPUT_CLASS = "class_map_lstm.npy"

IMG_SIZE = 128
MAX_FRAMES = 20
FLIP_HORIZONTAL = True  # Ubah False jika videonya terbalik (upside down), True jika mirror (kiri-kanan)

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,         # ✅ SUDAH DIPERBAIKI: Bisa detect 2 tangan
    min_detection_confidence=0.4, # Turunin dikit biar tangan yg gerak cepet tetep dapet
    min_tracking_confidence=0.4
)

def get_bbox_coordinates(frame, padding=20):
    """
    Mengembalikan koordinat (x_min, y_min, x_max, y_max) yang mencakup 
    SEMUA tangan yang terdeteksi (1 atau 2 tangan).
    """
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Inisialisasi titik min/max dengan kebalikan ukuran gambar
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        # Loop semua tangan yang ketemu (bisa 1 atau 2)
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y

        # Tambah padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Make Square (Biar rasio gak gepeng)
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        if box_w > box_h:
            diff = (box_w - box_h) // 2
            y_min = max(0, y_min - diff)
            y_max = min(h, y_max + diff)
        else:
            diff = (box_h - box_w) // 2
            x_min = max(0, x_min - diff)
            x_max = min(w, x_max + diff)
            
        return (x_min, y_min, x_max, y_max)

    return None

def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 5: 
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
    
    # ✅ VARIABEL BARU: Simpan KOORDINAT terakhir, bukan gambar terakhir
    last_bbox = None 

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue

        # 1. Flip jika perlu (Mirroring fix)
        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, -1)

        # 2. Cari koordinat tangan
        bbox = get_bbox_coordinates(frame)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            last_bbox = bbox # Update posisi terakhir
        
        elif last_bbox is not None:
            # ✅ LOGIKA ANTI-MATUNG:
            # Kalau tangan hilang, pakai koordinat LAMA untuk crop frame BARU.
            # Jadi videonya tetap gerak, cuma kotak fokusnya diem di tempat terakhir.
            x1, y1, x2, y2 = last_bbox
            
        else:
            # Kalau dari awal video gak ada tangan, ambil tengah
            h, w, _ = frame.shape
            min_dim = min(h, w)
            x1 = (w - min_dim) // 2
            y1 = (h - min_dim) // 2
            x2 = x1 + min_dim
            y2 = y1 + min_dim

        # 3. Lakukan Cropping berdasarkan koordinat
        crop = frame[y1:y2, x1:x2]
        
        # Validasi crop (kadang bisa error size 0)
        if crop.size == 0:
            crop = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) # Fallback
        else:
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frames.append(crop)

    cap.release()
    
    # Padding jika kurang frame
    if len(frames) > 0:
        while len(frames) < MAX_FRAMES:
            frames.append(frames[-1])
        return np.array(frames[:MAX_FRAMES])
    
    return None

def main():
    print("="*60)
    print("🚀 PRE-PROCESSING V2: 2 HANDS + ANTI-FREEZE")
    print("="*60)
    
    if not INPUT_DIR.exists():
        print(f"❌ Folder '{INPUT_DIR}' tidak ditemukan!")
        return

    # Hapus file lama biar gak numpuk/error
    if os.path.exists(OUTPUT_FILE_X): os.remove(OUTPUT_FILE_X)
    if os.path.exists(OUTPUT_FILE_Y): os.remove(OUTPUT_FILE_Y)

    classes = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
    class_map = {c: i for i, c in enumerate(classes)}
    
    X_data = []
    y_data = []
    
    print(f"📂 Memproses {len(classes)} kelas...")

    for cls in classes:
        cls_path = INPUT_DIR / cls
        videos = list(cls_path.glob("*.mp4")) + list(cls_path.glob("*.avi"))
        
        for vid in tqdm(videos, desc=f"Loading {cls}", leave=False):
            try:
                frames = process_video(vid)
                if frames is not None and len(frames) == MAX_FRAMES:
                    X_data.append(frames)
                    y_data.append(class_map[cls])
            except Exception as e:
                print(f"⚠️ Error {vid.name}: {e}")

    if not X_data:
        print("❌ Data kosong.")
        return

    X_data = np.array(X_data, dtype=np.uint8)
    y_data = np.array(y_data, dtype=np.uint8)
    
    print(f"\n💾 Menyimpan data baru... Shape: {X_data.shape}")
    np.save(OUTPUT_FILE_X, X_data)
    np.save(OUTPUT_FILE_Y, y_data)
    np.save(OUTPUT_CLASS, classes)
    print("✅ Selesai! Coba cek lagi pakai check_data.py")

main()