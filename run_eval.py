"""Run final evaluation on the best model and generate plots."""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.resnet_classifier import DeepfakeResNet
from src.preprocessing.dataset import create_dataloaders
from src.evaluation.metrics import full_evaluation, plot_roc_curve, plot_confusion_matrix

device = torch.device("cpu")

# Load best model
model = DeepfakeResNet(pretrained=False)
checkpoint = torch.load("models/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded. Best val AUC from training: {checkpoint.get('best_val_auc', 'N/A')}")

# Load validation data
_, val_loader = create_dataloaders(
    train_csv="data/splits/train.csv",
    val_csv="data/splits/val.csv",
    batch_size=16, num_workers=0, image_size=224
)

# Run evaluation
all_labels = []
all_probs = []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().flatten())

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

results = full_evaluation(all_labels, all_probs)
accuracy = (results['tp'] + results['tn']) / (results['tp'] + results['tn'] + results['fp'] + results['fn'])

print(f"\n{'='*50}")
print(f"  FINAL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"  AUC-ROC:           {results['auc_roc']:.4f}")
print(f"  Equal Error Rate:  {results['eer']:.4f}")
print(f"  Optimal Threshold: {results['eer_threshold']:.4f}")
print(f"  Accuracy:          {accuracy:.4f}")
print(f"  Precision:         {results['precision']:.4f}")
print(f"  Recall:            {results['recall']:.4f}")
print(f"  F1 Score:          {results['f1']:.4f}")
print(f"{'='*50}")

# Save plots
plots_dir = Path("models/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

plot_roc_curve(all_labels, all_probs, save_path=str(plots_dir / "roc_curve.png"))
print(f"\n  ROC curve saved to: {plots_dir / 'roc_curve.png'}")

preds = (all_probs >= results['eer_threshold']).astype(int)
plot_confusion_matrix(all_labels, preds, save_path=str(plots_dir / "confusion_matrix.png"))
print(f"  Confusion matrix saved to: {plots_dir / 'confusion_matrix.png'}")

print("\nDone! Ready for deployment.")
