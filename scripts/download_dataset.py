"""
Dataset Download & Preparation Script
======================================
Downloads freely available deepfake detection datasets from Kaggle
and prepares them for training.

Primary dataset: "Real and Fake Face Detection" by Ciplab (Yonsei University)
- ~1,900 images (960 real, 960 fake)
- Already cropped faces at 256x256
- ~500MB download
- No application required

Alternative: "140k Real and Fake Faces" (larger, ~2GB)
- 70k real (Flickr-Faces-HQ resized) + 70k fake (StyleGAN generated)

Usage:
    python scripts/download_dataset.py --dataset ciplab
    python scripts/download_dataset.py --dataset 140k --max-per-class 5000
"""

import os
import sys
import shutil
import argparse
import hashlib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_ciplab_dataset(data_dir: Path) -> Path:
    """
    Download 'Real and Fake Face Detection' dataset from Kaggle.
    
    Dataset structure after download:
        real_and_fake_face/
        ├── training_real/
        ├── training_fake/
        └── ...
    
    Returns:
        Path to the downloaded dataset directory.
    """
    import kagglehub
    
    print("=" * 60)
    print("Downloading: Real and Fake Face Detection (Ciplab)")
    print("Source: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection")
    print("Size: ~500MB")
    print("=" * 60)
    
    # kagglehub downloads to its cache directory
    dataset_path = kagglehub.dataset_download("ciplab/real-and-fake-face-detection")
    print(f"\n✅ Downloaded to: {dataset_path}")
    
    return Path(dataset_path)


def download_140k_dataset(data_dir: Path) -> Path:
    """
    Download '140k Real and Fake Faces' dataset from Kaggle.
    
    Returns:
        Path to the downloaded dataset directory.
    """
    import kagglehub
    
    print("=" * 60)
    print("Downloading: 140k Real and Fake Faces")
    print("Source: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
    print("Size: ~2GB (we'll use a subset)")
    print("=" * 60)
    
    dataset_path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
    print(f"\n✅ Downloaded to: {dataset_path}")
    
    return Path(dataset_path)


def organize_ciplab_dataset(source_dir: Path, output_dir: Path) -> tuple[list, list]:
    """
    Organize the Ciplab dataset into a standard structure.
    
    The Ciplab dataset has structure like:
        real_and_fake_face_detection/real_and_fake_face/training_real/...
        real_and_fake_face_detection/real_and_fake_face/training_fake/...
    
    We check the IMMEDIATE parent directory name (e.g. 'training_real',
    'training_fake') to classify images, not the full path.
    
    Returns:
        (list_of_real_paths, list_of_fake_paths)
    """
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    real_paths = []
    fake_paths = []
    
    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        # Use only the immediate parent directory name for classification
        parent_name = root_path.name.lower()
        
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            src = root_path / f
            
            # Check the immediate parent folder name for 'real' or 'fake'
            if 'fake' in parent_name:
                dst = fake_dir / f
                if not dst.exists():
                    shutil.copy2(src, dst)
                fake_paths.append(str(dst))
            elif 'real' in parent_name:
                dst = real_dir / f
                if not dst.exists():
                    shutil.copy2(src, dst)
                real_paths.append(str(dst))
    
    print(f"  Organized: {len(real_paths)} real, {len(fake_paths)} fake images")
    return real_paths, fake_paths


def organize_140k_dataset(source_dir: Path, output_dir: Path, 
                           max_per_class: int = 5000) -> tuple[list, list]:
    """
    Organize the 140k dataset into a standard structure, with optional subsampling.
    
    Returns:
        (list_of_real_paths, list_of_fake_paths)
    """
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    real_paths = []
    fake_paths = []
    
    # 140k dataset structure: real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/
    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        
        for f in sorted(files):  # Sort for reproducibility
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            src = root_path / f
            parent_name = root_path.name.lower()
            
            if parent_name == 'real' and len(real_paths) < max_per_class:
                dst = real_dir / f"{len(real_paths):06d}.jpg"
                if not dst.exists():
                    shutil.copy2(src, dst)
                real_paths.append(str(dst))
            elif parent_name == 'fake' and len(fake_paths) < max_per_class:
                dst = fake_dir / f"{len(fake_paths):06d}.jpg"
                if not dst.exists():
                    shutil.copy2(src, dst)
                fake_paths.append(str(dst))
            
            if len(real_paths) >= max_per_class and len(fake_paths) >= max_per_class:
                break
    
    print(f"  Organized: {len(real_paths)} real, {len(fake_paths)} fake images")
    return real_paths, fake_paths


def create_splits(real_paths: list, fake_paths: list, splits_dir: Path,
                  train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
    """
    Create train/val/test CSV split files.
    
    Args:
        real_paths: List of paths to real images.
        fake_paths: List of paths to fake images.
        splits_dir: Directory to save CSV files.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation. Test = 1 - train - val.
    """
    import random
    import csv
    
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine and shuffle
    all_samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    random.seed(42)  # Reproducible splits
    random.shuffle(all_samples)
    
    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_data = all_samples[:n_train]
    val_data = all_samples[n_train:n_train + n_val]
    test_data = all_samples[n_train + n_val:]
    
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        csv_path = splits_dir / f"{split_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            for path, label in split_data:
                writer.writerow([path, label])
        
        n_real = sum(1 for _, l in split_data if l == 0)
        n_fake = sum(1 for _, l in split_data if l == 1)
        print(f"  {split_name}: {len(split_data)} samples ({n_real} real, {n_fake} fake)")
    
    print(f"\n✅ Split CSVs saved to: {splits_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare deepfake detection dataset")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="ciplab",
        choices=["ciplab", "140k"],
        help="Which dataset to download: 'ciplab' (~500MB, 1.9k images) or '140k' (~2GB, subsettable)"
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5000,
        help="Max images per class for 140k dataset (default: 5000)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Base data directory"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio (default: 0.70)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    splits_dir = data_dir / "splits"
    
    print(f"\n📁 Data directory: {data_dir}")
    print(f"📁 Processed directory: {processed_dir}")
    print(f"📁 Splits directory: {splits_dir}\n")
    
    # Step 1: Download
    print("Step 1/3: Downloading dataset...")
    if args.dataset == "ciplab":
        source_dir = download_ciplab_dataset(data_dir)
    else:
        source_dir = download_140k_dataset(data_dir)
    
    # Step 2: Organize
    print("\nStep 2/3: Organizing images...")
    if args.dataset == "ciplab":
        real_paths, fake_paths = organize_ciplab_dataset(source_dir, processed_dir)
    else:
        real_paths, fake_paths = organize_140k_dataset(
            source_dir, processed_dir, max_per_class=args.max_per_class
        )
    
    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("\n❌ Error: No images found. Check the dataset structure.")
        print(f"   Source directory: {source_dir}")
        print("   Listing contents:")
        for item in source_dir.rglob("*"):
            if item.is_file():
                print(f"     {item.relative_to(source_dir)}")
                break
        sys.exit(1)
    
    # Step 3: Create splits
    print("\nStep 3/3: Creating train/val/test splits...")
    create_splits(real_paths, fake_paths, splits_dir, 
                  train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print(f"   Total images: {len(real_paths) + len(fake_paths)}")
    print(f"   Real: {len(real_paths)}")
    print(f"   Fake: {len(fake_paths)}")
    print(f"   Splits saved to: {splits_dir}")
    print("=" * 60)
    print(f"\nNext step: Run training with:")
    print(f"  python train.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
