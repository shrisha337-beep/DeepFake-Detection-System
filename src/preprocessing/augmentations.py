"""Data augmentation pipelines for deepfake detection.

Training pipeline applies aggressive augmentations that simulate
real-world image degradation (compression artefacts, blur, noise)
alongside standard geometric transforms.  Validation pipeline performs
only deterministic resize and normalisation.

All pipelines output PyTorch tensors with ImageNet normalisation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Build the training augmentation pipeline.

    The pipeline includes geometric, photometric, and compression-based
    augmentations designed to make the model robust to the varied quality
    of deepfake images encountered in the wild.

    Args:
        image_size: Target spatial resolution (square).

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=1.0,
        ),
        A.ImageCompression(
            quality_range=(30, 100),
            p=0.4,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.4,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Build the validation / inference augmentation pipeline.

    Only deterministic resize and ImageNet normalisation are applied —
    no stochastic augmentations.

    Args:
        image_size: Target spatial resolution (square).

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
