import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def augment_image(img):
    """Apply random augmentation to image"""
    # Random flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))
    
    # Random brightness
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)
    
    # Random contrast
    alpha = random.uniform(0.8, 1.2)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    
    # Random noise
    if random.random() > 0.7:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    
    return img

def augment_undersampled_classes(dataset_path, min_target=80):
    """
    Augmentasi otomatis untuk kelas yang kurang data
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Folder {dataset_path} tidak ditemukan!")
        return
    
    print("="*70)
    print("🔧 AUTO AUGMENTATION - Menambah Data Kelas yang Kurang")
    print("="*70)
    
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    for cls in classes:
        cls_path = dataset_path / cls
        files = list(cls_path.glob("*"))
        
        # Count frames
        total_frames = 0
        image_files = []
        
        for f in files:
            ext = f.suffix.lower()
            if ext in video_exts:
                total_frames += 15  # Estimate
            elif ext in image_exts:
                total_frames += 1
                image_files.append(f)
        
        # Check if needs augmentation
        if total_frames < min_target:
            needed = min_target - total_frames
            print(f"\n📸 Kelas '{cls}': {total_frames} frames (Kurang {needed})")
            
            if len(image_files) == 0:
                print(f"   ⚠️ Tidak ada foto untuk augmentasi. Skip.")
                continue
            
            # Augment images
            print(f"   🔄 Membuat {needed} augmented images...")
            
            for i in tqdm(range(needed), desc=f"   Augmenting {cls}", ncols=70):
                # Pick random image
                src_file = random.choice(image_files)
                img = cv2.imread(str(src_file))
                
                if img is None:
                    continue
                
                # Apply augmentation
                aug_img = augment_image(img)
                
                # Save augmented image
                output_name = f"{src_file.stem}_aug_{i:03d}{src_file.suffix}"
                output_path = cls_path / output_name
                cv2.imwrite(str(output_path), aug_img)
            
            print(f"   ✅ Selesai! Total sekarang: ~{min_target} frames")
        else:
            print(f"✅ Kelas '{cls}': {total_frames} frames (Sudah cukup)")
    
    print("\n" + "="*70)
    print("🎉 AUGMENTASI SELESAI!")
    print("="*70)
    print("💡 Langkah selanjutnya:")
    print("   1. Re-check: python check_dataset.py")
    print("   2. Training: python train.py")
    print("="*70)

if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).resolve().parent
    DATASET_PATH = CURRENT_DIR / "dataset"
    
    print(f"🔍 Dataset path: {DATASET_PATH}\n")
    
    if not DATASET_PATH.exists():
        print(f"❌ Folder dataset tidak ditemukan!")
        exit(1)
    
    # Set minimum target frames per class
    MIN_TARGET = 80
    
    print(f"🎯 Target minimum: {MIN_TARGET} frames per kelas")
    print(f"⚠️ Script ini akan membuat augmented images untuk kelas yang <{MIN_TARGET} frames")
    
    confirm = input("\n⚠️ Lanjutkan? (y/n): ").strip().lower()
    
    if confirm == 'y':
        augment_undersampled_classes(DATASET_PATH, MIN_TARGET)
    else:
        print("❌ Dibatalkan.")