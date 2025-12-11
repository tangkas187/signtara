import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json

# ==============================
# 🔧 Path Otomatis
# ==============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(CURRENT_DIR, "collectedvideos")
MODEL_DIR = os.path.join(CURRENT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# ⚙️ Parameter Dataset
# ==============================
FRAME_COUNT = 20   # ambil 20 frame pertama per video
IMG_SIZE = 64      # resize ke 64x64 pixel

def load_dataset():
    X, y = [], []
    class_names = sorted(os.listdir(VIDEO_DIR))
    print(f"📁 Ditemukan {len(class_names)} kelas: {class_names}")

    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(VIDEO_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        for video_file in os.listdir(class_path):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(class_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0

            while frame_count < FRAME_COUNT:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = frame / 255.0
                frames.append(frame)
                frame_count += 1

            cap.release()

            if len(frames) == FRAME_COUNT:
                X.append(np.array(frames))
                y.append(label_idx)
            else:
                print(f"⚠️ {video_file} di {class_name} cuma {len(frames)} frame, dilewati.")

    X = np.array(X)
    y = np.array(y)
    print(f"✅ Dataset selesai dimuat: {X.shape}, label: {len(y)}")
    return X, y, class_names

# ==============================
# 🧠 Model 3D CNN
# ==============================
def build_model(num_classes):
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=(FRAME_COUNT, IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling3D((1, 2, 2)),
        Dropout(0.25),

        Conv3D(64, (3, 3, 3), activation='relu'),
        MaxPooling3D((2, 2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==============================
# 🚀 Training
# ==============================
if __name__ == "__main__":
    print("📦 Memuat dataset dari:", VIDEO_DIR)
    X, y, class_names = load_dataset()

    y = to_categorical(y, num_classes=len(class_names))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Membangun model...")
    model = build_model(len(class_names))

    print("🏋️‍♂️ Melatih model...")
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    model_path = os.path.join(MODEL_DIR, "gesture_cnn.h5")
    model.save(model_path)
    print(f"✅ Model disimpan ke: {model_path}")

    with open(os.path.join(MODEL_DIR, "gesture_class_names.json"), "w") as f:
        json.dump(class_names, f)

    print("📄 Nama kelas disimpan ke gesture_class_names.json")
