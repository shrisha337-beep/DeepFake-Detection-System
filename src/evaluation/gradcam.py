"""Grad-CAM visualisation for deepfake detection models.

Provides two backends:
1. ``pytorch-grad-cam`` library (preferred, if installed).
2. A lightweight built-in implementation (fallback for environments
   where the library is unavailable, e.g. Python 3.14).

Both produce identical outputs: an RGB overlay and a grayscale activation
map highlighting the image regions that contribute most to the model's
prediction.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import pytorch-grad-cam; fall back to built-in implementation
# ---------------------------------------------------------------------------
_HAS_GRADCAM_LIB = False
try:
    from pytorch_grad_cam import GradCAM as _LibGradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image as _lib_show_cam
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    _HAS_GRADCAM_LIB = True
    logger.info("Using pytorch-grad-cam library for Grad-CAM.")
except ImportError:
    logger.info(
        "pytorch-grad-cam not available; using built-in Grad-CAM implementation."
    )


# ---------------------------------------------------------------------------
# Built-in Grad-CAM (no external dependency)
# ---------------------------------------------------------------------------

def _builtin_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
) -> np.ndarray:
    """Compute a Grad-CAM activation map without external libraries.

    Args:
        model: The neural network (must be in eval mode).
        input_tensor: Pre-processed input of shape ``(1, 3, H, W)``.
        target_layer: The convolutional layer to hook into.

    Returns:
        A ``float32`` ``(H_layer, W_layer)`` activation map in ``[0, 1]``.
    """
    activations = []
    gradients = []

    # Register hooks
    def _fwd_hook(_mod, _inp, out):
        activations.append(out.detach())

    def _bwd_hook(_mod, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(_fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        # Forward pass
        output = model(input_tensor)
        # Backward pass — target is the "fake" logit (single output)
        model.zero_grad()
        target = output.squeeze()
        target.backward()

        # Compute Grad-CAM
        act = activations[0].squeeze(0)   # (C, H, W)
        grad = gradients[0].squeeze(0)    # (C, H, W)

        # Global average pooling of gradients → channel weights
        weights = grad.mean(dim=(1, 2))   # (C,)

        # Weighted sum of activation maps
        cam = torch.zeros(act.shape[1:], dtype=act.dtype, device=act.device)
        for i, w in enumerate(weights):
            cam += w * act[i]

        # ReLU and normalise to [0, 1]
        cam = F.relu(cam)
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom

        return cam.cpu().numpy()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def _show_cam_on_image(
    img_float: np.ndarray, mask: np.ndarray, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Overlay a grayscale CAM onto an RGB image.

    Args:
        img_float: RGB image in ``[0, 1]`` float32, shape ``(H, W, 3)``.
        mask: Grayscale activation map in ``[0, 1]``, shape ``(H, W)``.
        colormap: OpenCV colormap to apply.

    Returns:
        Blended RGB uint8 image of shape ``(H, W, 3)``.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    blended = 0.5 * heatmap + 0.5 * img_float
    blended = np.clip(blended, 0, 1)
    return np.uint8(255 * blended)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image_np: np.ndarray,
    target_layer: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Grad-CAM heatmap for a single image.

    Args:
        model: A trained model with a ``get_gradcam_target_layer`` method
            (e.g. :class:`DeepfakeResNet`).  Will be set to ``eval`` mode
            internally.
        input_tensor: Pre-processed input tensor of shape ``(1, 3, H, W)``
            or ``(3, H, W)`` (a batch dimension is added automatically).
        original_image_np: The original RGB image as a ``uint8`` numpy
            array of shape ``(H, W, 3)``, used for the overlay.
        target_layer: The convolutional layer to compute Grad-CAM on.
            If ``None``, ``model.get_gradcam_target_layer()`` is used.

    Returns:
        A tuple ``(visualization_rgb, grayscale_cam)`` where:
        - ``visualization_rgb`` is a ``uint8`` ``(H, W, 3)`` RGB overlay.
        - ``grayscale_cam`` is a ``float32`` ``(H, W)`` activation map
          in ``[0, 1]``.
    """
    model.eval()

    # Ensure 4-D tensor
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Resolve target layer
    if target_layer is None:
        if hasattr(model, "get_gradcam_target_layer"):
            target_layer = model.get_gradcam_target_layer()
        else:
            raise ValueError(
                "target_layer is None and model has no "
                "'get_gradcam_target_layer' method."
            )

    # Normalise original image to [0, 1] float for overlay
    if original_image_np.dtype == np.uint8:
        rgb_float = original_image_np.astype(np.float32) / 255.0
    else:
        rgb_float = original_image_np.astype(np.float32)
        if rgb_float.max() > 1.0:
            rgb_float /= 255.0

    # Resize to match input tensor spatial dims
    _, _, h, w = input_tensor.shape
    rgb_resized = cv2.resize(rgb_float, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- Choose backend ---
    if _HAS_GRADCAM_LIB:
        targets = [BinaryClassifierOutputTarget(1)]  # explain the "fake" class
        with _LibGradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # (H, W)
        visualization = _lib_show_cam(rgb_resized, grayscale_cam, use_rgb=True)
    else:
        # Built-in fallback
        input_tensor.requires_grad_(True)
        grayscale_cam = _builtin_gradcam(model, input_tensor, target_layer)
        # Resize CAM to match image dimensions
        grayscale_cam = cv2.resize(grayscale_cam, (w, h), interpolation=cv2.INTER_LINEAR)
        visualization = _show_cam_on_image(rgb_resized, grayscale_cam)

    return visualization, grayscale_cam


def create_gradcam_comparison(
    model: nn.Module,
    image_path: str,
    transform,
    save_path: Optional[str] = None,
) -> None:
    """Create a side-by-side comparison of original image and Grad-CAM overlay.

    Args:
        model: A trained deepfake detection model.
        image_path: Path to the source image file.
        transform: An Albumentations pipeline that produces the model's
            expected input (should include Normalize + ToTensorV2).
        save_path: If provided, save the comparison figure to this path.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load original image
    original_pil = Image.open(image_path).convert("RGB")
    original_np = np.array(original_pil, dtype=np.uint8)

    # Apply transform to get model input
    transformed = transform(image=original_np)
    input_tensor: torch.Tensor = transformed["image"].unsqueeze(0)  # (1, 3, H, W)

    # Generate Grad-CAM
    visualization, grayscale_cam = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        original_image_np=original_np,
    )

    # Get prediction
    model.eval()
    with torch.no_grad():
        logit = model(input_tensor).squeeze()
        prob = torch.sigmoid(logit).item()
    pred_label = "FAKE" if prob >= 0.5 else "REAL"

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original_np)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(visualization)
    axes[1].set_title(
        f"Grad-CAM Overlay — {pred_label} ({prob:.3f})", fontsize=13,
    )
    axes[1].axis("off")

    fig.suptitle(f"Deepfake Detection: {image_path.name}", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path_obj), dpi=150, bbox_inches="tight")
        logger.info("Grad-CAM comparison saved to %s.", save_path)

    plt.close(fig)
