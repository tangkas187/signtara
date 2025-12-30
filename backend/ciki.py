import torch
import sys

print("="*40)
print("🚀 CEK GPU PYTORCH (TANPA INSTALL MANUAL)")
print("="*40)

# Ini logika yang kamu mau:
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print(f"✅ GPU Ditemukan: {torch.cuda.get_device_name(0)}")
    print("👉 CUDA Version:", torch.version.cuda)
    print("✨ Siap training tanpa ribet install driver!")
else:
    print("❌ GPU tidak terdeteksi. Cek installasi pip-nya.")