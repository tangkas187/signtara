import numpy as np
import cv2
import sys
import random

# Konfigurasi
N_SAMPLES = 5  # Mau cek berapa video acak?
DELAY_MS = 150 # Kecepatan putar (makin besar makin lambat)

def main():
    print("📂 Sedang meload file .npy (tunggu sebentar)...")
    try:
        X = np.load("X_video.npy")
        y = np.load("y_video.npy")
        classes = np.load("class_map_lstm.npy", allow_pickle=True)
    except FileNotFoundError:
        print("❌ File .npy gak ketemu! Pastikan ada di folder yang sama.")
        sys.exit()

    print(f"✅ Data Terbaca: {len(X)} sampel.")
    print(f"👉 Tekan 'ESC' atau 'q' di keyboard untuk stop.\n")

    # Acak urutan biar kita gak cuma liat data awal doang
    indices = list(range(len(X)))
    random.shuffle(indices)

    for i in indices[:N_SAMPLES]:
        video_data = X[i]  # Shape: (20, 128, 128, 3)
        label_idx = y[i]
        label_name = classes[label_idx]

        print(f"🎬 Memutar Video Label: [ {label_name} ]")

        # Loop per frame
        for frame_idx, frame in enumerate(video_data):
            # 1. Convert balik ke BGR (karena di npy tersimpan RGB)
            # OpenCV butuh BGR buat display, kalau gak nanti kulitnya jadi biru (kayak Avatar)
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 2. Perbesar biar enak dilihat (128px itu kecil banget)
            display_frame = cv2.resize(display_frame, (400, 400), interpolation=cv2.INTER_NEAREST)

            # 3. Tambah info text di layar
            cv2.putText(display_frame, f"{label_name} | Frame {frame_idx+1}/20", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Cek Data NPY", display_frame)

            # Tunggu tombol
            key = cv2.waitKey(DELAY_MS) & 0xFF
            if key == ord('q') or key == 27: # q atau ESC
                print("👋 Keluar...")
                cv2.destroyAllWindows()
                sys.exit()
        
        # Jeda dikit antar video
        cv2.waitKey(1000)

    cv2.destroyAllWindows()
    print("✅ Selesai cek sampel.")

if __name__ == "__main__":
    main()