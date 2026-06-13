"""Video frame extraction utilities for deepfake detection.

Provides two strategies:
- ``extract_frames``: Write every *N*-th frame to disk as JPEG.
- ``sample_frames_from_video``: Uniformly sample *N* frames into memory
  (no disk I/O) for fast inference pipelines.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    every_n: int = 10,
) -> List[str]:
    """Extract every *N*-th frame from a video and save as JPEG.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory to write extracted frames into.
            Created automatically if it does not exist.
        every_n: Interval between extracted frames.  ``1`` means every
            frame, ``10`` means every 10th frame, etc.

    Returns:
        A list of absolute file paths to the saved JPEG frames.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the video cannot be opened by OpenCV.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    frame_idx = 0
    video_stem = video_path.stem

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % every_n == 0:
                filename = f"{video_stem}_frame_{frame_idx:06d}.jpg"
                save_path = str(output_dir / filename)
                cv2.imwrite(save_path, frame)
                saved_paths.append(save_path)
            frame_idx += 1
    finally:
        cap.release()

    logger.info(
        "Extracted %d frames from %s (every_n=%d, total_frames=%d).",
        len(saved_paths), video_path.name, every_n, frame_idx,
    )
    return saved_paths


def sample_frames_from_video(
    video_path: str,
    num_frames: int = 15,
) -> List[np.ndarray]:
    """Uniformly sample *N* frames from a video without saving to disk.

    Frames are returned as RGB ``uint8`` numpy arrays suitable for direct
    input to face extractors or augmentation pipelines.

    Args:
        video_path: Path to the source video file.
        num_frames: Number of frames to sample.  If the video has fewer
            frames, all available frames are returned.

    Returns:
        A list of RGB numpy arrays, each of shape ``(H, W, 3)``.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the video cannot be opened by OpenCV.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback: read all frames sequentially
        frames: List[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

        if len(frames) == 0:
            logger.warning("No frames read from %s.", video_path)
            return []

        # Subsample uniformly
        if len(frames) <= num_frames:
            return frames
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]

    # Compute evenly spaced frame indices
    actual_num = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, actual_num, dtype=int)

    sampled_frames: List[np.ndarray] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                logger.debug("Failed to read frame %d from %s.", idx, video_path)
                continue
            sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    logger.info(
        "Sampled %d / %d frames from %s.",
        len(sampled_frames), total_frames, video_path.name,
    )
    return sampled_frames
