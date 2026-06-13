"""Deploy to Hugging Face Spaces."""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "Shri04/deepfake-detector"
REPO_TYPE = "space"
SPACE_SDK = "gradio"

PROJECT_ROOT = Path(__file__).resolve().parent

# Files and directories to upload
UPLOAD_ITEMS = [
    # Core files
    "README.md",
    "requirements_hf.txt",
    # Frontend (entry point)
    "frontend/app.py",
    # API module
    "api/__init__.py",
    "api/inference.py",
    "api/schemas.py",
    # Source modules
    "src/__init__.py",
    "src/model/__init__.py",
    "src/model/resnet_classifier.py",
    "src/preprocessing/__init__.py",
    "src/preprocessing/face_extractor.py",
    "src/preprocessing/augmentations.py",
    "src/preprocessing/dataset.py",
    "src/evaluation/__init__.py",
    "src/evaluation/gradcam.py",
    "src/evaluation/metrics.py",
    "src/utils/__init__.py",
    "src/utils/video_utils.py",
    # Config
    "configs/config.yaml",
    # Model weights
    "models/best_model.pth",
]


def main():
    api = HfApi()

    # Step 1: Create the Space repo
    print(f"Creating Space: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            space_sdk=SPACE_SDK,
            exist_ok=True,
            private=False,
        )
        print(f"  [OK] Space created/exists: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"  [WARN] create_repo: {e}")

    # Step 1.5: Clean up old files in remote space repo
    print("Cleaning up old remote files...")
    try:
        api.delete_file(
            path_in_repo="requirements_hf.txt",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print("  [OK] Removed remote requirements_hf.txt")
    except Exception as e:
        # Ignore if file does not exist
        pass

    # Step 2: Upload files
    print(f"\nUploading {len(UPLOAD_ITEMS)} files...")
    for rel_path in UPLOAD_ITEMS:
        local_path = PROJECT_ROOT / rel_path
        if not local_path.exists():
            print(f"  [SKIP] {rel_path} (not found)")
            continue

        path_in_repo = "requirements.txt" if rel_path == "requirements_hf.txt" else rel_path
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {rel_path} -> {path_in_repo} ({size_mb:.1f} MB)...", end=" ", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"Deployment complete!")
    print(f"  Space URL: https://huggingface.co/spaces/{REPO_ID}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
