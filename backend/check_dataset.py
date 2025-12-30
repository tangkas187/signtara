import os
from pathlib import Path
from collections import defaultdict

def count_dataset_files(dataset_path):
    """
    Hitung jumlah video dan foto per kelas dalam dataset
    Berikan rekomendasi untuk balance dataset
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Folder {dataset_path} tidak ditemukan!")
        return
    
    print("="*70)
    print("📊 DATASET BALANCE CHECKER")
    print("="*70)
    
    stats = defaultdict(lambda: {"videos": 0, "images": 0, "total_frames": 0})
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]
    
    # Hitung file per kelas
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    for cls in classes:
        cls_path = dataset_path / cls
        files = list(cls_path.glob("*"))
        
        for f in files:
            ext = f.suffix.lower()
            if ext in video_exts:
                stats[cls]["videos"] += 1
                stats[cls]["total_frames"] += 15  # Estimasi 15 frame per video
            elif ext in image_exts:
                stats[cls]["images"] += 1
                stats[cls]["total_frames"] += 1
    
    # Tampilkan hasil
    print(f"\nDitemukan {len(classes)} kelas:\n")
    print(f"{'Kelas':<15} {'Video':>7} {'Foto':>7} {'Est.Frames':>12} {'Status':>12}")
    print("-"*70)
    
    total_videos = 0
    total_images = 0
    min_frames = float('inf')
    max_frames = 0
    
    for cls in sorted(stats.keys()):
        v = stats[cls]["videos"]
        i = stats[cls]["images"]
        f = stats[cls]["total_frames"]
        
        total_videos += v
        total_images += i
        min_frames = min(min_frames, f)
        max_frames = max(max_frames, f)
        
        # Status
        if f < 80:
            status = "⚠️ Kurang"
        elif f > 300:
            status = "⚠️ Terlalu Banyak"
        else:
            status = "✅ Bagus"
        
        print(f"{cls:<15} {v:>7} {i:>7} {f:>12} {status:>12}")
    
    print("-"*70)
    print(f"{'TOTAL':<15} {total_videos:>7} {total_images:>7}")
    print()
    
    # Analisis & Rekomendasi
    print("="*70)
    print("📋 ANALISIS & REKOMENDASI")
    print("="*70)
    
    print(f"\n📊 Statistik Dataset:")
    print(f"   • Total Kelas: {len(classes)}")
    print(f"   • Total Video: {total_videos}")
    print(f"   • Total Foto: {total_images}")
    print(f"   • Frame Terendah: {min_frames}")
    print(f"   • Frame Tertinggi: {max_frames}")
    print(f"   • Rasio Imbalance: {max_frames/min_frames if min_frames > 0 else 'N/A'}x")
    
    # Evaluasi balance
    avg_frames = sum(s["total_frames"] for s in stats.values()) / len(stats)
    imbalance_ratio = max_frames / min_frames if min_frames > 0 else 0
    
    print(f"\n🎯 Target Setelah Balancing:")
    print(f"   • Setiap kelas → 120 samples (otomatis oleh train.py)")
    
    print(f"\n💡 REKOMENDASI:")
    
    if imbalance_ratio > 5:
        print(f"   ⚠️ IMBALANCE TINGGI ({imbalance_ratio:.1f}x)")
        print(f"   → Kelas dengan <80 frames akan di-oversample (duplikasi)")
        print(f"   → Kelas dengan >300 frames akan di-undersample (dibuang)")
        print(f"   → SOLUSI: Tambah data ke kelas yang kurang")
    elif imbalance_ratio > 3:
        print(f"   ⚠️ IMBALANCE SEDANG ({imbalance_ratio:.1f}x)")
        print(f"   → Balancing otomatis akan bekerja, tapi akurasi bisa terpengaruh")
        print(f"   → SOLUSI OPSIONAL: Tambah data ke kelas yang kurang")
    else:
        print(f"   ✅ BALANCE BAGUS ({imbalance_ratio:.1f}x)")
        print(f"   → Dataset sudah cukup balance untuk training")
    
    print(f"\n🎬 Rekomendasi Komposisi per Kelas:")
    print(f"   • IDEAL: 10-15 video + 30-50 foto = ~180-275 frames")
    print(f"   • MINIMUM: 80 frames (agar tidak terlalu banyak oversample)")
    print(f"   • MAKSIMUM: 300 frames (agar tidak banyak data terbuang)")
    
    # Rekomendasi spesifik per kelas
    print(f"\n🔧 Rekomendasi Spesifik:")
    for cls in sorted(stats.keys()):
        f = stats[cls]["total_frames"]
        v = stats[cls]["videos"]
        i = stats[cls]["images"]
        
        if f < 80:
            needed_videos = max(0, (80 - f) // 15)
            needed_images = max(0, 80 - f - (needed_videos * 15))
            print(f"   • {cls}: KURANG!")
            print(f"     → Tambah {needed_videos} video ATAU {needed_images} foto")
        elif f > 300:
            excess = f - 250
            print(f"   • {cls}: TERLALU BANYAK!")
            print(f"     → {excess} frames akan dibuang saat balancing")
            print(f"     → Tidak masalah, tapi bisa kurangi jika mau")
    
    print("\n" + "="*70)
    print("💾 CARA GUNAKAN:")
    print("="*70)
    print("1. Simpan script ini sebagai 'check_dataset.py'")
    print("2. Jalankan: python check_dataset.py")
    print("3. Ikuti rekomendasi untuk balance dataset")
    print("4. Training ulang dengan: python train.py")
    print("="*70)

if __name__ == "__main__":
    # Path ke dataset Anda (otomatis deteksi)
    CURRENT_DIR = Path(__file__).resolve().parent
    DATASET_PATH = CURRENT_DIR / "dataset"
    
    print(f"🔍 Checking dataset di: {DATASET_PATH}\n")
    
    if not DATASET_PATH.exists():
        print(f"❌ Folder dataset tidak ditemukan!")
        print(f"📁 Pastikan struktur folder:")
        print(f"   backend/")
        print(f"   ├── check_dataset.py  ← Script ini")
        print(f"   ├── train.py")
        print(f"   └── dataset/")
        print(f"       ├── A/")
        print(f"       ├── B/")
        print(f"       ├── C/")
        print(f"       ├── Cinta/")
        print(f"       └── ...")
        exit(1)
    
    count_dataset_files(DATASET_PATH)