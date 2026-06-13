"""
Training Entry Point
====================
Main script to train the deepfake detection model.

Usage:
    python train.py
    python train.py --config configs/config.yaml
    python train.py --epochs 20 --batch-size 8 --lr 0.00005
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.resnet_classifier import DeepfakeResNet
from src.preprocessing.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.evaluation.metrics import full_evaluation, plot_roc_curve, plot_confusion_matrix


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with command-line arguments."""
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.image_size is not None:
        config['data']['image_size'] = args.image_size
    return config


def print_training_info(config: dict, model: DeepfakeResNet, device: torch.device) -> None:
    """Print training configuration summary."""
    params = model.count_parameters()
    
    print("\n" + "=" * 60)
    print("🧠 DEEPFAKE DETECTION - Training Session")
    print("=" * 60)
    print(f"  Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device:         {device}")
    print(f"  Architecture:   {config['model']['architecture']}")
    print(f"  Pretrained:     {config['model']['pretrained']}")
    print(f"  Image size:     {config['data']['image_size']}x{config['data']['image_size']}")
    print(f"  Batch size:     {config['data']['batch_size']}")
    print(f"  Epochs:         {config['training']['epochs']}")
    print(f"  Learning rate:  {config['training']['learning_rate']}")
    print(f"  Weight decay:   {config['training']['weight_decay']}")
    print(f"  Scheduler:      {config['training']['scheduler']}")
    print(f"  Label smooth:   {config['training']['label_smoothing']}")
    print(f"  Dropout:        {config['model']['dropout']}")
    print("-" * 60)
    print(f"  Total params:     {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")
    print(f"  Frozen params:    {params['frozen']:,}")
    print(f"  Trainable ratio:  {params['trainable']/params['total']*100:.1f}%")
    print("=" * 60 + "\n")


def run_final_evaluation(model: DeepfakeResNet, val_loader, device: torch.device,
                          save_dir: Path) -> None:
    """Run comprehensive evaluation on the validation set and save plots."""
    print("\n" + "=" * 60)
    print("📊 Running Final Evaluation...")
    print("=" * 60)
    
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute all metrics
    results = full_evaluation(all_labels, all_probs)
    
    print(f"\n  AUC-ROC:               {results['auc_roc']:.4f}")
    print(f"  Equal Error Rate:      {results['eer']:.4f}")
    print(f"  Optimal Threshold:     {results['eer_threshold']:.4f}")
    accuracy = (results['tp'] + results['tn']) / max(results['tp'] + results['tn'] + results['fp'] + results['fn'], 1)
    print(f"  Accuracy:              {accuracy:.4f}")
    print(f"  Precision:             {results['precision']:.4f}")
    print(f"  Recall:                {results['recall']:.4f}")
    print(f"  F1 Score:              {results['f1']:.4f}")
    
    # Save plots
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_roc_curve(all_labels, all_probs, save_path=str(plots_dir / "roc_curve.png"))
    print(f"  ROC curve saved to:    {plots_dir / 'roc_curve.png'}")
    
    preds = (all_probs >= results['eer_threshold']).astype(int)
    plot_confusion_matrix(all_labels, preds, save_path=str(plots_dir / "confusion_matrix.png"))
    print(f"  Confusion matrix:      {plots_dir / 'confusion_matrix.png'}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override image size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    config = override_config(config, args)
    
    # Device setup (CPU for Intel Iris Xe)
    device = torch.device("cpu")
    
    # Verify data exists
    train_csv = Path(config['data']['train_csv'])
    val_csv = Path(config['data']['val_csv'])
    
    if not train_csv.exists():
        print(f"❌ Training CSV not found: {train_csv}")
        print(f"   Run 'python scripts/download_dataset.py' first to download and prepare data.")
        sys.exit(1)
    
    if not val_csv.exists():
        print(f"❌ Validation CSV not found: {val_csv}")
        sys.exit(1)
    
    # Create data loaders
    print("📦 Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
    )
    print(f"   Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    
    # Create model
    print("\n🔨 Building model...")
    model = DeepfakeResNet(
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
    )
    model.to(device)
    
    # Print training info
    print_training_info(config, model, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer config — flatten nested YAML config to match Trainer's expected keys
    trainer_config = {
        "lr": config['training']['learning_rate'],
        "weight_decay": config['training']['weight_decay'],
        "loss": "label_smoothing" if config['training'].get('label_smoothing', 0) > 0 else "bce",
        "loss_kwargs": {"smoothing": config['training'].get('label_smoothing', 0.1)},
        "scheduler_T_max": config['training']['epochs'],
        "early_stopping_patience": config['training']['early_stopping_patience'],
        "checkpoint_dir": config['logging']['save_dir'],
        "checkpoint_name": "best_model.pth",
        "unfreeze_schedule": {
            int(k): v for k, v in config['training'].get('unfreeze_schedule', {}).items()
        },
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_dict=trainer_config,
        device=device,
    )
    
    # Train
    print("🚀 Starting training...\n")
    history = trainer.fit(num_epochs=config['training']['epochs'])
    
    # Load best model for final evaluation
    best_model_path = Path(config['logging']['save_dir']) / "best_model.pth"
    if best_model_path.exists():
        print(f"\n📂 Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    run_final_evaluation(model, val_loader, device, Path(config['logging']['save_dir']))
    
    print("\n✅ Training complete!")
    print(f"   Best model saved to: {best_model_path}")
    print(f"\nNext steps:")
    print(f"  1. Run API:     uvicorn api.main:app --reload")
    print(f"  2. Run Gradio:  python frontend/app.py")


if __name__ == "__main__":
    main()
