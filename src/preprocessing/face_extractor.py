"""Face extraction utilities for deepfake detection preprocessing.

Provides two extraction backends:
- MTCNN (facenet_pytorch): High-quality offline face extraction with landmarks.
- OpenCV DNN: Fast real-time face detection using a pre-trained Caffe model.

Both methods detect, crop (with margin), and resize faces to the target size.
"""

import logging
import math
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

logger = logging.getLogger(__name__)

# OpenCV DNN Caffe model URLs (official OpenCV GitHub)
_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
    "deploy.prototxt"
)
_CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Default model cache directory
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "opencv_face_detector"


class FaceExtractor:
    """Detects and extracts face regions from images or video frames.

    Attributes:
        face_size: Target output size (square) for extracted face crops.
        margin: Pixel margin to add around the detected bounding box.
        device: Torch device string (always ``'cpu'`` for this project).
    """

    def __init__(
        self,
        face_size: int = 224,
        margin: int = 40,
        device: str = "cpu",
    ) -> None:
        """Initialise FaceExtractor.

        Args:
            face_size: Side length of the square output face crop.
            margin: Extra pixels to include around the detected bounding box.
            device: Torch device string. Must be ``'cpu'`` (no CUDA support).
        """
        self.face_size = face_size
        self.margin = margin
        self.device = torch.device(device)

        # Lazily initialised backends
        self._mtcnn: Optional[MTCNN] = None
        self._opencv_net: Optional[cv2.dnn.Net] = None

    # ------------------------------------------------------------------
    # Convenience method (used by inference.py)
    # ------------------------------------------------------------------

    def extract(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extract the primary face from an RGB numpy image.

        This is a convenience wrapper that converts RGB → BGR and
        delegates to :meth:`extract_face_opencv` for fast inference.

        Args:
            image_rgb: An ``(H, W, 3)`` RGB uint8 numpy array.

        Returns:
            A ``(face_size, face_size, 3)`` RGB uint8 numpy array, or
            ``None`` if no face is detected.
        """
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return self.extract_face_opencv(bgr)

    # ------------------------------------------------------------------
    # MTCNN backend (high-quality, offline preprocessing)
    # ------------------------------------------------------------------

    def _get_mtcnn(self) -> MTCNN:
        """Lazy-load the MTCNN detector."""
        if self._mtcnn is None:
            self._mtcnn = MTCNN(
                image_size=self.face_size,
                margin=self.margin,
                keep_all=True,
                post_process=False,  # return raw uint8 images
                device=self.device,
            )
            logger.info("MTCNN detector initialised on %s.", self.device)
        return self._mtcnn

    def extract_face_mtcnn(self, image_path: str) -> Optional[np.ndarray]:
        """Extract the largest face from an image using MTCNN.

        Args:
            image_path: Path to the source image file.

        Returns:
            A ``(face_size, face_size, 3)`` RGB uint8 numpy array of the
            largest detected face, or ``None`` if no face is found.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, IOError) as exc:
            logger.warning("Cannot open image %s: %s", image_path, exc)
            return None

        mtcnn = self._get_mtcnn()
        boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

        if boxes is None or len(boxes) == 0:
            logger.debug("No face detected in %s.", image_path)
            return None

        # Select the largest face by bounding-box area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = int(np.argmax(areas))
        box = boxes[best_idx]

        # Crop with margin, clamping to image bounds
        img_w, img_h = image.size
        x1 = max(0, int(box[0]) - self.margin)
        y1 = max(0, int(box[1]) - self.margin)
        x2 = min(img_w, int(box[2]) + self.margin)
        y2 = min(img_h, int(box[3]) + self.margin)

        face_crop = image.crop((x1, y1, x2, y2)).resize(
            (self.face_size, self.face_size), Image.BILINEAR,
        )
        face_array = np.array(face_crop, dtype=np.uint8)

        # Optionally align using landmarks if available
        if landmarks is not None and landmarks[best_idx] is not None:
            lm = landmarks[best_idx]
            # landmarks order: left_eye, right_eye, nose, mouth_left, mouth_right
            left_eye = tuple(lm[0].astype(int))
            right_eye = tuple(lm[1].astype(int))
            img_np = np.array(image, dtype=np.uint8)
            aligned = self.align_face(img_np, left_eye, right_eye)
            if aligned is not None:
                # Re-crop aligned image
                aligned_pil = Image.fromarray(aligned)
                a_w, a_h = aligned_pil.size
                cx1 = max(0, int(box[0]) - self.margin)
                cy1 = max(0, int(box[1]) - self.margin)
                cx2 = min(a_w, int(box[2]) + self.margin)
                cy2 = min(a_h, int(box[3]) + self.margin)
                face_crop = aligned_pil.crop((cx1, cy1, cx2, cy2)).resize(
                    (self.face_size, self.face_size), Image.BILINEAR,
                )
                face_array = np.array(face_crop, dtype=np.uint8)

        logger.debug(
            "Extracted face from %s (box=%s, prob=%.3f).",
            image_path, box, probs[best_idx],
        )
        return face_array

    # ------------------------------------------------------------------
    # OpenCV DNN backend (fast, real-time)
    # ------------------------------------------------------------------

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        """Download a file if it does not already exist."""
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s → %s ...", url, dest)
        urllib.request.urlretrieve(url, str(dest))
        logger.info("Download complete: %s", dest)

    def _get_opencv_net(self) -> cv2.dnn.Net:
        """Lazy-load the OpenCV DNN face detector."""
        if self._opencv_net is None:
            prototxt_path = _MODEL_DIR / "deploy.prototxt"
            caffemodel_path = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

            self._download_file(_PROTOTXT_URL, prototxt_path)
            self._download_file(_CAFFEMODEL_URL, caffemodel_path)

            self._opencv_net = cv2.dnn.readNetFromCaffe(
                str(prototxt_path), str(caffemodel_path),
            )
            logger.info("OpenCV DNN face detector loaded.")
        return self._opencv_net

    def extract_face_opencv(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> Optional[np.ndarray]:
        """Extract the largest face from a BGR frame using OpenCV DNN.

        Args:
            frame: A BGR image as a numpy array (typical OpenCV format).
            confidence_threshold: Minimum detection confidence to accept.

        Returns:
            A ``(face_size, face_size, 3)`` RGB uint8 numpy array of the
            largest detected face, or ``None`` if no face is found.
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame.")
            return None

        net = self._get_opencv_net()
        h, w = frame.shape[:2]

        # Prepare the blob (300×300, mean subtraction)
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 177.0, 123.0), swapRB=False, crop=False,
        )
        net.setInput(blob)
        detections = net.forward()

        # Collect valid detections
        best_box: Optional[Tuple[int, int, int, int]] = None
        best_area = 0
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < confidence_threshold:
                continue
            x1 = max(0, int(detections[0, 0, i, 3] * w) - self.margin)
            y1 = max(0, int(detections[0, 0, i, 4] * h) - self.margin)
            x2 = min(w, int(detections[0, 0, i, 5] * w) + self.margin)
            y2 = min(h, int(detections[0, 0, i, 6] * h) + self.margin)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            logger.debug("No face detected by OpenCV DNN.")
            return None

        x1, y1, x2, y2 = best_box
        face_crop = frame[y1:y2, x1:x2]
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(
            face_crop_rgb, (self.face_size, self.face_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return face_resized.astype(np.uint8)

    # ------------------------------------------------------------------
    # Face alignment via eye landmarks
    # ------------------------------------------------------------------

    @staticmethod
    def align_face(
        image: np.ndarray,
        left_eye: Tuple[int, int],
        right_eye: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Align a face in-place using the positions of both eyes.

        The image is rotated so that the line between the eyes is
        horizontal, which can improve recognition/classification accuracy.

        Args:
            image: RGB uint8 numpy array of the full image.
            left_eye: ``(x, y)`` pixel coordinates of the left eye centre.
            right_eye: ``(x, y)`` pixel coordinates of the right eye centre.

        Returns:
            The rotated RGB image as a numpy array, or ``None`` on failure.
        """
        try:
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = math.degrees(math.atan2(dy, dx))

            # Rotation centre is the midpoint between the eyes
            cx = (left_eye[0] + right_eye[0]) // 2
            cy = (left_eye[1] + right_eye[1]) // 2

            h, w = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)
            aligned = cv2.warpAffine(
                image, rotation_matrix, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned
        except Exception as exc:
            logger.warning("Face alignment failed: %s", exc)
            return None
