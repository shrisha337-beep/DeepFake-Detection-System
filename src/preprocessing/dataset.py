"""PyTorch Dataset and DataLoader utilities for deepfake detection.

Expects a CSV file with at least two columns:
    - ``image_path``: absolute or relative path to the image file.
    - ``label``: integer label (0 = real, 1 = fake).
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.augmentations import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """Map-style dataset that loads face images from paths listed in a CSV.

    Each sample is a tuple ``(image_tensor, label)`` where
    ``image_tensor`` is a float32 tensor of shape ``(3, H, W)`` and
    ``label`` is an ``int`` (0 = real, 1 = fake).
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            csv_path: Path to a CSV with ``image_path`` and ``label`` columns.
            transform: An Albumentations ``Compose`` pipeline (or any
                callable that accepts ``image=np.ndarray`` and returns a
                dict with an ``"image"`` key).
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        required_cols = {"image_path", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found: {list(self.df.columns)}"
            )

        self.transform = transform
        logger.info(
            "DeepfakeDataset loaded %d samples from %s "
            "(real=%d, fake=%d).",
            len(self.df),
            csv_path,
            int((self.df["label"] == 0).sum()),
            int((self.df["label"] == 1).sum()),
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return a single (image_tensor, label) pair.

        Args:
            idx: Sample index.

        Returns:
            A tuple of ``(image_tensor, label)`` where ``image_tensor`` is
            a ``float32`` tensor and ``label`` is an ``int``.

        Raises:
            RuntimeError: If the image cannot be loaded or transformed.
        """
        row = self.df.iloc[idx]
        image_path = str(row["image_path"])
        label = int(row["label"])

        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image, dtype=np.uint8)
        except (OSError, IOError) as exc:
            raise RuntimeError(
                f"Failed to load image at index {idx}: {image_path}"
            ) from exc

        if self.transform is not None:
            try:
                transformed = self.transform(image=image_np)
                image_tensor: torch.Tensor = transformed["image"]
            except Exception as exc:
                raise RuntimeError(
                    f"Transform failed for image at index {idx}: {image_path}"
                ) from exc
        else:
            # Fallback: simple conversion without augmentation
            image_tensor = torch.from_numpy(
                image_np.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return image_tensor, label


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation ``DataLoader`` instances.

    Args:
        train_csv: Path to the training CSV file.
        val_csv: Path to the validation CSV file.
        batch_size: Mini-batch size for both loaders.
        num_workers: Number of data-loading worker processes.
            Use ``0`` for single-process loading (safest on Windows / CPU).
        image_size: Target spatial resolution for augmentations.

    Returns:
        A tuple ``(train_loader, val_loader)``.
    """
    train_transform = get_train_transforms(image_size=image_size)
    val_transform = get_val_transforms(image_size=image_size)

    train_dataset = DeepfakeDataset(csv_path=train_csv, transform=train_transform)
    val_dataset = DeepfakeDataset(csv_path=val_csv, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # CPU-only, no need to pin
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    logger.info(
        "DataLoaders created — train: %d batches, val: %d batches (bs=%d).",
        len(train_loader), len(val_loader), batch_size,
    )
    return train_loader, val_loader
