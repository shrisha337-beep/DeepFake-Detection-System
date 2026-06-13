"""Training loop for the deepfake detection model.

The ``Trainer`` class orchestrates single-epoch training, validation,
progressive layer unfreezing, early stopping, and model checkpointing.

This module is designed for **CPU-only** training — no CUDA, no mixed
precision.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import get_loss_function

logger = logging.getLogger(__name__)

# Default configuration — callers can override any key.
_DEFAULT_CONFIG: Dict[str, Any] = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "loss": "bce",
    "loss_kwargs": {},
    "scheduler_T_max": 10,
    "early_stopping_patience": 5,
    "checkpoint_dir": "models",
    "checkpoint_name": "best_model.pth",
    # Progressive unfreezing schedule: {epoch_number: layer_name_to_unfreeze}
    # e.g. {3: "layer3", 6: "layer4"}
    "unfreeze_schedule": {},
}


class Trainer:
    """Manages the full training lifecycle for :class:`DeepfakeResNet`.

    Attributes:
        model: The neural network to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Merged configuration dictionary.
        device: Torch device (CPU).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config_dict: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> None:
        """Initialise the Trainer.

        Args:
            model: A ``DeepfakeResNet`` (or compatible) model.
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
            config_dict: Configuration overrides.  Keys not provided
                fall back to ``_DEFAULT_CONFIG``.
            device: Torch device string (must be ``'cpu'``).
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Merge user config with defaults
        self.config: Dict[str, Any] = {**_DEFAULT_CONFIG, **(config_dict or {})}

        # Optimiser — only parameters that require gradients
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        # Learning-rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["scheduler_T_max"],
        )

        # Loss function
        self.criterion = get_loss_function(
            self.config["loss"], **self.config.get("loss_kwargs", {}),
        )

        # Tracking
        self.best_val_auc: float = 0.0
        self.epochs_without_improvement: int = 0
        self.history: List[Dict[str, Any]] = []

        logger.info("Trainer initialised on device=%s.", self.device)

    # ------------------------------------------------------------------
    # Single-epoch training
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> Tuple[float, float]:
        """Run one full pass over the training set.

        Returns:
            A tuple ``(average_loss, accuracy)`` for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc="  Train",
            leave=False,
            dynamic_ncols=True,
        )
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(images).squeeze(1)  # (B,)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float, List[int], List[float]]:
        """Evaluate the model on the validation set.

        Returns:
            A tuple ``(avg_loss, accuracy, auc_roc, all_labels, all_probs)``.
            ``auc_roc`` is ``0.0`` if computation fails (e.g. single-class
            batch).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels: List[int] = []
        all_probs: List[float] = []

        pbar = tqdm(
            self.val_loader,
            desc="  Val  ",
            leave=False,
            dynamic_ncols=True,
        )
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).float()

            logits = self.model(images).squeeze(1)
            loss = self.criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += images.size(0)

            all_labels.extend(labels.cpu().int().tolist())
            all_probs.extend(probs.cpu().tolist())

        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        # AUC-ROC (requires both classes present)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            logger.warning("AUC computation failed (single class in val set?).")
            auc = 0.0

        return avg_loss, accuracy, auc, all_labels, all_probs

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int) -> List[Dict[str, Any]]:
        """Train for *num_epochs* with unfreezing, early stopping, and checkpointing.

        Args:
            num_epochs: Total number of epochs to train.

        Returns:
            A list of per-epoch metric dictionaries.
        """
        patience = self.config["early_stopping_patience"]
        unfreeze_schedule: Dict[int, str] = self.config.get("unfreeze_schedule", {})

        logger.info("Starting training for %d epochs.", num_epochs)
        print(f"\n{'='*70}")
        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>9} | {'Val Acc':>8} | {'Val AUC':>8} | {'LR':>10}")
        print(f"{'='*70}")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # --- Progressive unfreezing ---
            if epoch in unfreeze_schedule:
                layer = unfreeze_schedule[epoch]
                logger.info("Epoch %d: unfreezing '%s'.", epoch, layer)
                self.model.unfreeze_layer(layer)
                # Re-create optimizer to include newly unfrozen params
                self.optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.config["lr"],
                    weight_decay=self.config["weight_decay"],
                )
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, num_epochs - epoch),
                )

            # --- Train & validate ---
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, val_auc, val_labels, val_probs = self.validate()
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - epoch_start

            # --- Epoch summary ---
            print(
                f"{epoch:>6d} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
                f"{val_loss:>9.4f} | {val_acc:>8.4f} | {val_auc:>8.4f} | "
                f"{current_lr:>10.2e}  ({elapsed:.1f}s)"
            )

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "lr": current_lr,
                "elapsed_s": elapsed,
            }
            self.history.append(epoch_metrics)

            # --- Checkpointing (best val AUC) ---
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.epochs_without_improvement = 0
                ckpt_dir = Path(self.config["checkpoint_dir"])
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / self.config["checkpoint_name"]
                self.save_checkpoint(str(ckpt_path))
                print(f"       ✓ New best val AUC: {val_auc:.4f} — saved to {ckpt_path}")
            else:
                self.epochs_without_improvement += 1

            # --- Early stopping ---
            if self.epochs_without_improvement >= patience:
                print(
                    f"\n⏹  Early stopping after {patience} epochs "
                    f"without improvement (best AUC={self.best_val_auc:.4f})."
                )
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        print(f"{'='*70}")
        print(f"Training complete. Best val AUC: {self.best_val_auc:.4f}\n")
        return self.history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint to disk.

        The checkpoint includes model weights, optimiser state, scheduler
        state, best validation AUC, and training history.

        Args:
            path: File path to write the checkpoint to.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_auc": self.best_val_auc,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
            "config": self.config,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s.", path)

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint from disk and resume state.

        Args:
            path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        self.history = checkpoint.get("history", [])

        logger.info(
            "Checkpoint loaded from %s (best_val_auc=%.4f).",
            path, self.best_val_auc,
        )
