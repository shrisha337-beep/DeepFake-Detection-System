"""ResNet-50 based binary classifier for deepfake detection.

Architecture overview:
    ResNet-50 backbone (conv1 → layer4)  →  Global Average Pooling
    → Dropout → Linear(2048, 512) → ReLU → BN → Dropout → Linear(512, 1)

Early layers (conv1, bn1, layer1, layer2) are frozen by default and can
be progressively unfrozen during training for fine-tuning.
"""

import logging
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)


class DeepfakeResNet(nn.Module):
    """ResNet-50 with a custom binary classification head.

    Attributes:
        backbone: The ResNet-50 feature extractor (``fc`` replaced with
            ``Identity``).
        classifier: Custom classification head producing a single logit.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model.

        Args:
            pretrained: If ``True``, load ImageNet-pretrained weights for
                the ResNet-50 backbone.
            dropout: Dropout probability for the first dropout layer in
                the classifier head.  The second layer uses ``dropout / 2``.
        """
        super().__init__()

        # ----- backbone -----
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Replace the original fully-connected layer with Identity so
        # ``forward`` returns the 2048-d feature vector after avg-pool.
        self.backbone.fc = nn.Identity()

        # ----- freeze early layers -----
        self._freeze_layers(["conv1", "bn1", "layer1", "layer2"])

        # ----- classifier head -----
        self.classifier = nn.Sequential(
            OrderedDict([
                ("drop1", nn.Dropout(p=dropout)),
                ("fc1", nn.Linear(2048, 512)),
                ("relu", nn.ReLU(inplace=True)),
                ("bn", nn.BatchNorm1d(512)),
                ("drop2", nn.Dropout(p=dropout / 2)),
                ("fc2", nn.Linear(512, 1)),
            ])
        )

        # Initialise classifier weights
        self._init_classifier()

        logger.info(
            "DeepfakeResNet created (pretrained=%s, dropout=%.2f). %s",
            pretrained, dropout, self.count_parameters(),
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_classifier(self) -> None:
        """Apply Kaiming initialisation to the classifier head."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _freeze_layers(self, layer_prefixes: list) -> None:
        """Freeze all parameters whose name starts with any given prefix."""
        for name, param in self.backbone.named_parameters():
            for prefix in layer_prefixes:
                if name.startswith(prefix):
                    param.requires_grad = False
                    break

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw logits for the input batch.

        Args:
            x: Input tensor of shape ``(B, 3, 224, 224)``.

        Returns:
            Logits tensor of shape ``(B, 1)``.  Apply ``torch.sigmoid``
            to obtain probabilities; use ``BCEWithLogitsLoss`` for
            training (numerically stable).
        """
        features = self.backbone(x)  # (B, 2048)
        logits = self.classifier(features)  # (B, 1)
        return logits

    # ------------------------------------------------------------------
    # Progressive unfreezing
    # ------------------------------------------------------------------

    def unfreeze_layer(self, layer_name: str) -> None:
        """Unfreeze all backbone parameters whose name starts with *layer_name*.

        Args:
            layer_name: Prefix to match, e.g. ``"layer3"`` or ``"layer2"``.
        """
        count = 0
        for name, param in self.backbone.named_parameters():
            if name.startswith(layer_name):
                param.requires_grad = True
                count += 1
        logger.info("Unfroze %d parameters matching '%s'.", count, layer_name)

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter in the entire model."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen.")

    # ------------------------------------------------------------------
    # Grad-CAM support
    # ------------------------------------------------------------------

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualisation.

        Returns:
            The last bottleneck block of ``layer4``.
        """
        return self.backbone.layer4[-1]

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def count_parameters(self) -> Dict[str, int]:
        """Count total, trainable, and frozen parameters.

        Returns:
            A dict with keys ``"total"``, ``"trainable"``, ``"frozen"``.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}
