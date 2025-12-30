import cv2
import os
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import csv

# =========================
# 🔧 KONFIGURASI
# =========================
INPUT_DIR = Path("dataset_dynamic")
OUTPUT_CSV = "dataset_keypoints.csv"

MAX_FRAMES = 20      # Jumlah frame per video
NUM_HANDS = 2        # Maksimal tangan yang dideteksi
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3 # x, y, z

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_hand_landmarks(frame):
    """
    Ekstrak koordinat (x, y, z) untuk 2 tangan.
    Output: List flat sepanjang (2 tangan * 21 titik * 3 xyz) = 126 nilai
    """
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Wadah untuk data tangan (default 0 semua jika tidak ada tangan)
    # Urutan: [Tangan1_x, Tangan1_y, ..., Tangan2_z]
    frame_data = np.zeros(NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK)

    if results.multi_hand_landmarks:
        # Loop detected hands (maksimal 2)
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= NUM_HANDS: break # Amankan jika terdeteksi > 2
            
            # Ambil 21 landmark
            for j, lm in enumerate(hand_landmarks.landmark):
                # Hitung index flattening
                idx = (i * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK) + (j * COORDS_PER_LANDMARK)
                
                # Simpan x, y, z (Normalisasi 0-1 dari MediaPipe udah oke)
                frame_data[idx]     = lm.x
                frame_data[idx + 1] = lm.y
                frame_data[idx + 2] = lm.z 
                
    return frame_data.tolist()

def process_video_to_landmarks(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 5:
        cap.release()
        return None

    # Ambil index frame rata (sampling)
    indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
    
    video_sequence = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: 
            # Jika frame rusak, isi dengan nol (padding darurat)
            video_sequence.extend([0.0] * (NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK))
            continue

        # Extract Landmark
        landmarks = get_hand_landmarks(frame)
        video_sequence.extend(landmarks) # Flatten frame ke dalam baris video

    cap.release()
    
    # Validasi panjang data
    expected_len = MAX_FRAMES * NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
    
    # Padding jika kurang frame (copy frame terakhir)
    current_len = len(video_sequence)
    if current_len > 0 and current_len < expected_len:
        frame_len = NUM_HANDS * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
        last_frame_data = video_sequence[-frame_len:]
        while len(video_sequence) < expected_len:
            video_sequence.extend(last_frame_data)
            
    return video_sequence

def main():
    print("="*60)
    print("🚀 PRE-PROCESSING CSV: LANDMARK EXTRACTION")
    print("="*60)
    
    if not INPUT_DIR.exists():
        print(f"❌ Folder '{INPUT_DIR}' tidak ditemukan!")
        return

    classes = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()])
    print(f"📂 Kelas ditemukan: {classes}")

    # Siapkan Header CSV
    # Format: label, frame1_h1_0_x, frame1_h1_0_y, ..., frame20_h2_20_z
    header = ["label"]
    for f in range(MAX_FRAMES):
        for h in range(NUM_HANDS):
            for l in range(LANDMARKS_PER_HAND):
                header.append(f"f{f}_h{h}_l{l}_x")
                header.append(f"f{f}_h{h}_l{l}_y")
                header.append(f"f{f}_h{h}_l{l}_z")
                
    print(f"📝 Membuat {OUTPUT_CSV} dengan {len(header)} kolom...")

    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for cls in classes:
            cls_path = INPUT_DIR / cls
            videos = list(cls_path.glob("*.mp4")) + list(cls_path.glob("*.avi"))
            
            for vid in tqdm(videos, desc=f"Processing {cls}"):
                try:
                    # Proses video jadi 1 baris panjang
                    sequence_data = process_video_to_landmarks(vid)
                    
                    if sequence_data and len(sequence_data) == (len(header) - 1):
                        # Tulis [Label, data...]
                        row = [cls] + sequence_data
                        writer.writerow(row)
                    else:
                        pass # Video gagal/korup
                        
                except Exception as e:
                    print(f"⚠️ Error {vid.name}: {e}")

    print(f"\n✅ Selesai! Data tersimpan di '{OUTPUT_CSV}'")
    print("👉 Selanjutnya: Buat file training baru (bukan CNN) tapi murni LSTM/Dense.")

if __name__ == "__main__":
    main()