"""Deepfake detection inference engine.

Provides the ``DeepfakePredictor`` class which wraps model loading,
face extraction, image/video preprocessing, and prediction into a
single, easy-to-use interface.  All inference runs on CPU.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``src.*`` imports resolve
# regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.model.resnet_classifier import DeepfakeResNet
from src.preprocessing.face_extractor import FaceExtractor
from src.preprocessing.augmentations import get_val_transforms

logger = logging.getLogger(__name__)


class DeepfakePredictor:
    """End-to-end deepfake predictor for images and videos.

    Parameters
    ----------
    model_path : str
        Path to a saved ``DeepfakeResNet`` checkpoint (``.pth`` file).
    device : str, optional
        Torch device string.  Defaults to ``"cpu"``.
    """

    # Confidence threshold from config (evaluation.threshold)
    THRESHOLD: float = 0.5

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model_path = model_path
        self.is_loaded: bool = False

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        self.model = DeepfakeResNet()

        if os.path.isfile(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                # Support both raw state_dict and wrapped checkpoint dicts
                state_dict = (
                    checkpoint.get("model_state_dict", checkpoint)
                    if isinstance(checkpoint, dict)
                    else checkpoint
                )
                self.model.load_state_dict(state_dict)
                self.is_loaded = True
                logger.info("Model loaded successfully from %s", model_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load model weights from %s: %s. "
                    "Running with randomly initialized weights.",
                    model_path,
                    exc,
                )
        else:
            logger.warning(
                "Model file not found at %s. Predictor will run with "
                "randomly initialized weights (is_loaded=False).",
                model_path,
            )

        self.model.to(self.device)
        self.model.eval()

        # ------------------------------------------------------------------
        # Face extractor & transforms
        # ------------------------------------------------------------------
        self.face_extractor = FaceExtractor()
        self.transforms = get_val_transforms()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(self, image_bytes: bytes) -> Optional[dict]:
        """Run deepfake prediction on raw image bytes.

        Parameters
        ----------
        image_bytes : bytes
            Raw bytes of a JPEG / PNG / WebP image.

        Returns
        -------
        dict or None
            A dict compatible with ``PredictionResponse`` if a face was
            detected, otherwise ``None``.
        """
        # Decode bytes → numpy BGR
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.error("Failed to decode image bytes.")
            return None

        # Convert BGR → RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.predict_numpy(rgb)

    def predict_numpy(self, image_np: np.ndarray) -> Optional[dict]:
        """Run deepfake prediction on a NumPy RGB image.

        Parameters
        ----------
        image_np : np.ndarray
            HxWx3 NumPy array in **RGB** colour order, dtype ``uint8``.

        Returns
        -------
        dict or None
            A dict compatible with ``PredictionResponse`` if a face was
            detected, otherwise ``None``.
        """
        # Extract face
        face = self.face_extractor.extract(image_np)
        if face is None:
            logger.info("No face detected in the provided image.")
            return None

        # Ensure face is uint8 RGB for transforms
        if face.dtype != np.uint8:
            face = face.astype(np.uint8)

        # Apply validation transforms (Albumentations expects HWC numpy)
        transformed = self.transforms(image=face)
        tensor = transformed["image"]  # CxHxW torch.Tensor

        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logit = self.model(tensor)
            # Model outputs raw logit for binary classification (num_classes=1)
            fake_prob = torch.sigmoid(logit).item()

        prediction = "FAKE" if fake_prob >= self.THRESHOLD else "REAL"
        confidence = fake_prob if prediction == "FAKE" else 1.0 - fake_prob

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "face_detected": True,
        }

    def predict_video(
        self, video_path: str, num_frames: int = 15
    ) -> dict:
        """Run deepfake prediction on a video file.

        Uniformly samples ``num_frames`` frames from the video, extracts
        faces, runs prediction on each, and returns an aggregated result.

        Parameters
        ----------
        video_path : str
            Path to a video file (MP4 recommended).
        num_frames : int, optional
            Number of frames to sample.  Defaults to ``15``.

        Returns
        -------
        dict
            A dict compatible with ``VideoResponse``.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video file: %s", video_path)
            return self._empty_video_response()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            logger.error("Video has no frames: %s", video_path)
            return self._empty_video_response()

        # Compute uniform sample indices
        num_frames = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

        frame_predictions: list[dict] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, bgr = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = self.predict_numpy(rgb)
            if result is not None:
                frame_predictions.append({
                    "frame_index": int(idx),
                    "fake_probability": result["fake_probability"],
                })

        cap.release()

        if not frame_predictions:
            return self._empty_video_response()

        # Aggregate
        probs = [fp["fake_probability"] for fp in frame_predictions]
        avg_prob = float(np.mean(probs))
        prediction = "FAKE" if avg_prob >= self.THRESHOLD else "REAL"
        confidence = avg_prob if prediction == "FAKE" else 1.0 - avg_prob

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "avg_fake_probability": round(avg_prob, 4),
            "num_frames_analyzed": len(frame_predictions),
            "frame_predictions": frame_predictions,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_video_response() -> dict:
        """Return a neutral ``VideoResponse``-compatible dict when no
        frames could be analysed."""
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "avg_fake_probability": 0.0,
            "num_frames_analyzed": 0,
            "frame_predictions": [],
        }
