"""Quick smoke test for all project modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

print("Testing module imports...")

# 1. Model
from src.model.resnet_classifier import DeepfakeResNet
model = DeepfakeResNet(pretrained=True)
params = model.count_parameters()
print(f"  [OK] DeepfakeResNet: {params['total']:,} params ({params['trainable']:,} trainable)")

# 2. Preprocessing
from src.preprocessing.augmentations import get_train_transforms, get_val_transforms
train_t = get_train_transforms()
val_t = get_val_transforms()
print(f"  [OK] Augmentations: train={len(train_t)}, val={len(val_t)} transforms")

# 3. Dataset
from src.preprocessing.dataset import DeepfakeDataset
print(f"  [OK] DeepfakeDataset class loaded")

# 4. Losses
from src.training.losses import get_loss_function
bce = get_loss_function("bce")
focal = get_loss_function("focal")
smooth = get_loss_function("label_smoothing")
print(f"  [OK] Loss functions: bce, focal, label_smoothing")

# 5. Metrics
from src.evaluation.metrics import full_evaluation
print(f"  [OK] Metrics module loaded")

# 6. Grad-CAM
from src.evaluation.gradcam import generate_gradcam
print(f"  [OK] Grad-CAM module loaded (built-in fallback)")

# 7. Video utils
from src.utils.video_utils import extract_frames, sample_frames_from_video
print(f"  [OK] Video utils loaded")

# 8. Forward pass test
import torch
dummy = torch.randn(1, 3, 224, 224)
model.eval()
with torch.no_grad():
    out = model(dummy)
prob = torch.sigmoid(out).item()
print(f"  [OK] Forward pass: logit={out.item():.4f}, prob={prob:.4f}")

# 9. API imports
from api.inference import DeepfakePredictor
from api.schemas import PredictionResponse, VideoResponse
print(f"  [OK] API modules loaded")

print("\n=== ALL TESTS PASSED ===")
