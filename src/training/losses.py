"""Loss functions for deepfake detection training.

Provides:
- ``FocalLoss``: Down-weights easy examples so the model focuses on hard
  or misclassified samples.  Particularly useful with class imbalance.
- ``LabelSmoothingBCELoss``: Applies label smoothing to binary targets
  before computing BCE, acting as a regulariser.
- ``get_loss_function``: Factory helper that returns the requested loss
  by name.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for binary classification (logit inputs).

    Reference: *Lin et al.*, "Focal Loss for Dense Object Detection", 2017.

    The loss is defined as::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where ``p_t`` is the model's estimated probability for the true class.

    Attributes:
        alpha: Balancing factor for the positive class.
        gamma: Focusing parameter — higher values down-weight easy examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """Initialise FocalLoss.

        Args:
            alpha: Weight for the positive class.  The negative class
                receives weight ``1 - alpha``.
            gamma: Focusing exponent.  ``gamma = 0`` recovers standard
                weighted BCE.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs of shape ``(B,)`` or ``(B, 1)``.
            targets: Ground-truth labels of the same shape as *logits*,
                with values in ``{0.0, 1.0}``.

        Returns:
            Scalar loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable BCE per-element
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )

        probs = torch.sigmoid(logits)
        # p_t: probability of the *true* class
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        # alpha_t: per-sample balancing weight
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        return loss.mean()


class LabelSmoothingBCELoss(nn.Module):
    """Binary cross-entropy with label smoothing.

    Hard labels ``{0, 1}`` are softened to ``{smoothing/2, 1 - smoothing/2}``
    before computing BCE loss.  This prevents the model from becoming
    over-confident and acts as a mild regulariser.

    Attributes:
        smoothing: Total smoothing amount (split symmetrically).
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        """Initialise LabelSmoothingBCELoss.

        Args:
            smoothing: Smoothing factor in ``[0, 1)``.  ``0`` recovers
                standard BCE.
        """
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed BCE loss.

        Args:
            logits: Raw model outputs of shape ``(B,)`` or ``(B, 1)``.
            targets: Ground-truth labels of the same shape as *logits*,
                with values in ``{0.0, 1.0}``.

        Returns:
            Scalar loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Smooth the labels
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        loss = F.binary_cross_entropy_with_logits(
            logits, smoothed_targets, reduction="mean",
        )
        return loss


def get_loss_function(name: str = "bce", **kwargs: Any) -> nn.Module:
    """Factory function that returns a loss module by name.

    Args:
        name: One of ``"bce"``, ``"focal"``, ``"label_smoothing"``.
        **kwargs: Extra keyword arguments forwarded to the loss
            constructor.

    Returns:
        An ``nn.Module`` loss function ready for training.

    Raises:
        ValueError: If *name* is not recognised.
    """
    name = name.lower().strip()

    if name == "bce":
        loss_fn = nn.BCEWithLogitsLoss(**kwargs)
    elif name == "focal":
        loss_fn = FocalLoss(**kwargs)
    elif name in ("label_smoothing", "label_smooth", "ls"):
        loss_fn = LabelSmoothingBCELoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss function '{name}'. "
            f"Choose from: 'bce', 'focal', 'label_smoothing'."
        )

    logger.info("Loss function: %s(%s)", type(loss_fn).__name__, kwargs)
    return loss_fn
