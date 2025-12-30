import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# =========================
# 🔧 KONFIGURASI
# =========================
IMG_SIZE = 64        # Samakan dengan prepo
MAX_FRAMES = 20
BATCH_SIZE = 8       # Sesuaikan dengan VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Using Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

# =========================
# 1. DATASET CLASS (Pengganti Generator)
# =========================
class VideoDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Ambil video: (Frames, H, W, Channels)
        video = self.X[idx]
        label = self.y[idx]

        # Normalisasi manual (0-255 -> 0.0-1.0)
        video = video.astype(np.float32) / 255.0

        # PyTorch butuh urutan: (Frames, Channels, H, W)
        # Input asli: (Frames, H, W, C) -> (20, 64, 64, 3)
        # Output: (20, 3, 64, 64)
        video = torch.tensor(video).permute(0, 3, 1, 2)
        
        # Augmentasi (Opsional, simpel dulu biar jalan)
        # Kalau mau augmentasi video di PyTorch agak kompleks, 
        # jadi kita skip dulu biar kodingan gak error.
        
        return video, torch.tensor(label, dtype=torch.long)

# =========================
# 2. MODEL ARSITEKTUR (LRCN)
# =========================
class LRCN(nn.Module):
    def __init__(self, num_classes, hidden_size=64):
        super(LRCN, self).__init__()
        
        # --- A. CNN (Mata) ---
        # Load MobileNetV2 pre-trained
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        self.base_cnn = models.mobilenet_v2(weights=weights)
        
        # Hapus classifier bawaan MobileNet (kita cuma butuh fiturnya)
        # MobileNetV2 features output channelnya 1280
        self.base_cnn.classifier = nn.Identity() 
        self.base_cnn_features = self.base_cnn.features # Ambil bagian feature extractor aja
        
        # Pooling (Global Average Pooling ala PyTorch)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # --- B. LSTM (Otak) ---
        # Input size 1280 (output dari MobileNetV2)
        self.lstm = nn.LSTM(input_size=1280, hidden_size=hidden_size, 
                            num_layers=1, batch_first=True)
        
        # --- C. Classifier ---
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes) 
        # Note: Tidak perlu Softmax di akhir, karena CrossEntropyLoss PyTorch sudah include Softmax

    def forward(self, x):
        # x shape: (Batch, Frames, Channel, H, W)
        batch_size, time_steps, C, H, W = x.size()
        
        # 1. CNN Processing
        # Kita gabung Batch & Frames biar bisa diproses CNN sekaligus
        # Shape jadi: (Batch * Frames, C, H, W)
        c_in = x.view(batch_size * time_steps, C, H, W)
        
        # Masuk CNN
        c_out = self.base_cnn_features(c_in) # Output: (B*T, 1280, h, w)
        c_out = self.gap(c_out)              # Output: (B*T, 1280, 1, 1)
        c_out = c_out.view(c_out.size(0), -1) # Flatten: (B*T, 1280)
        
        # 2. LSTM Processing
        # Balikin shape ke (Batch, Frames, Features)
        r_in = c_out.view(batch_size, time_steps, -1)
        
        # Masuk LSTM
        # r_out shape: (Batch, Frames, Hidden)
        # h_n shape: (1, Batch, Hidden) -> Last hidden state
        _, (h_n, _) = self.lstm(r_in)
        
        # Ambil hidden state terakhir (representasi video)
        # Output shape: (Batch, Hidden)
        lstm_out = h_n[-1] 
        
        # 3. Classifier
        x = self.dropout(lstm_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# =========================
# 3. FUNGSI TRAINING
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for videos, labels in tqdm(loader, desc="Training", leave=False):
        videos, labels = videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Validation", leave=False):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(loader), 100 * correct / total

# =========================
# 4. MAIN PROGRAM
# =========================
def main():
    CURRENT_DIR = Path(__file__).resolve().parent
    MODELS_DIR = CURRENT_DIR / "models_torch"
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Load Data ---
    try:
        print("📦 Memuat data .npy...")
        X = np.load("X_video.npy")
        y = np.load("y_video.npy")
        classes = np.load("class_map_lstm.npy", allow_pickle=True)
        print(f"✅ Data Loaded: {len(X)} sampel | {len(classes)} kelas")
    except FileNotFoundError:
        print("❌ Data tidak ditemukan! Pastikan file .npy ada.")
        sys.exit(1)

    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Buat DataLoader
    train_ds = VideoDataset(X_train, y_train)
    val_ds = VideoDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Init Model
    model = LRCN(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # 🚀 PHASE 1: WARM UP (CNN FROZEN)
    # ==========================================
    print("\n❄️ PHASE 1: Training LSTM (CNN Beku)...")
    
    # Bekukan CNN
    for param in model.base_cnn_features.parameters():
        param.requires_grad = False
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_acc = 0.0

    for epoch in range(10): # 10 Epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/10 | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # ==========================================
    # 🚀 PHASE 2: FINE TUNING (CNN UNFREEZE)
    # ==========================================
    print("\n🔥 PHASE 2: Fine Tuning (CNN Unfreeze)...")
    
    # Unfreeze semua
    for param in model.base_cnn_features.parameters():
        param.requires_grad = True
        
    # Bekukan lagi 40% layer awal (biar fitur dasar image net gak rusak)
    # Di PyTorch MobileNetV2, 'features' adalah list layer. Kita bekukan index 0-10.
    for i, child in enumerate(model.base_cnn_features.children()):
        if i < 10: 
            for param in child.parameters():
                param.requires_grad = False

    # Learning Rate Kecil
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    for epoch in range(15): # 40 Epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/40 (FT) | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "bisindo_lstm_best.pth")
            print(f"   💾 Model saved! (New Best Acc: {best_acc:.2f}%)")

    # Save Metadata
    with open(MODELS_DIR / "class_names.json", "w") as f:
        json.dump(classes.tolist(), f)

    print("\n🎉 Training Selesai! Model disimpan di folder 'models_torch'")

if __name__ == "__main__":
    main()