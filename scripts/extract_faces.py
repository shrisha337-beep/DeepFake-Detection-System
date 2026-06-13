"""
Face Extraction Preprocessing Script
======================================
Extract faces from raw images/videos using MTCNN or OpenCV DNN.
Run this if your dataset contains full frames (not pre-cropped faces).

For pre-cropped datasets (like Ciplab or 140k), this step is NOT needed —
the download_dataset.py script handles everything.

Usage:
    python scripts/extract_faces.py --input data/raw/videos --output data/processed
    python scripts/extract_faces.py --input data/raw/images --output data/processed --detector mtcnn
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.face_extractor import FaceExtractor
from src.utils.video_utils import extract_frames
import cv2


def extract_faces_from_images(input_dir: Path, output_dir: Path,
                                extractor: FaceExtractor, 
                                label: str = "unknown") -> list[str]:
    """Extract faces from a directory of images."""
    output_label_dir = output_dir / label
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in input_dir.rglob("*") 
        if f.suffix.lower() in image_extensions
    ]
    
    saved_paths = []
    skipped = 0
    
    for img_path in tqdm(image_files, desc=f"Extracting faces ({label})"):
        face = extractor.extract_face_mtcnn(str(img_path))
        
        if face is not None:
            save_name = f"{img_path.stem}_face.jpg"
            save_path = output_label_dir / save_name
            # Convert RGB to BGR for OpenCV saving
            cv2.imwrite(str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            saved_paths.append(str(save_path))
        else:
            skipped += 1
    
    print(f"  {label}: {len(saved_paths)} faces extracted, {skipped} skipped (no face found)")
    return saved_paths


def extract_faces_from_videos(input_dir: Path, output_dir: Path,
                                extractor: FaceExtractor,
                                label: str = "unknown",
                                every_n: int = 10) -> list[str]:
    """Extract faces from a directory of videos."""
    output_label_dir = output_dir / label
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp dir for frames
    temp_frames_dir = output_dir / "_temp_frames"
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = [
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in video_extensions
    ]
    
    saved_paths = []
    
    for video_path in tqdm(video_files, desc=f"Processing videos ({label})"):
        # Extract frames
        frame_dir = temp_frames_dir / video_path.stem
        frame_dir.mkdir(exist_ok=True)
        
        frame_paths = extract_frames(str(video_path), str(frame_dir), every_n=every_n)
        
        # Extract faces from frames
        for frame_path in frame_paths:
            face = extractor.extract_face_mtcnn(frame_path)
            if face is not None:
                save_name = f"{video_path.stem}_{Path(frame_path).stem}_face.jpg"
                save_path = output_label_dir / save_name
                cv2.imwrite(str(save_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                saved_paths.append(str(save_path))
        
        # Clean up temp frames
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)
    
    # Clean up temp dir
    import shutil
    shutil.rmtree(temp_frames_dir, ignore_errors=True)
    
    print(f"  {label}: {len(saved_paths)} faces extracted from {len(video_files)} videos")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Extract faces from images/videos")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with images or videos")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for cropped faces")
    parser.add_argument("--label", type=str, default="unknown",
                        help="Label for the images (e.g., 'real' or 'fake')")
    parser.add_argument("--detector", type=str, default="mtcnn",
                        choices=["mtcnn", "opencv"],
                        help="Face detector to use")
    parser.add_argument("--mode", type=str, default="images",
                        choices=["images", "videos"],
                        help="Whether input contains images or videos")
    parser.add_argument("--every-n", type=int, default=10,
                        help="For videos: extract every Nth frame")
    parser.add_argument("--face-size", type=int, default=224,
                        help="Output face size (default: 224)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    print(f"📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"🔍 Detector: {args.detector}")
    print(f"📏 Face size: {args.face_size}x{args.face_size}")
    
    extractor = FaceExtractor(face_size=args.face_size, device="cpu")
    
    if args.mode == "images":
        extract_faces_from_images(input_dir, output_dir, extractor, label=args.label)
    else:
        extract_faces_from_videos(input_dir, output_dir, extractor, 
                                   label=args.label, every_n=args.every_n)
    
    print("\n✅ Face extraction complete!")


if __name__ == "__main__":
    main()
