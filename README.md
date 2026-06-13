---
title: Deepfake Detector
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
app_file: frontend/app.py
pinned: true
license: mit
tags:
  - deepfake-detection
  - computer-vision
  - resnet50
  - transfer-learning
  - gradio
---

# 🛡️ Deepfake Detector

**Real-time deepfake detection using a fine-tuned ResNet-50 model.**

## Model Performance

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.9424** |
| **Accuracy** | **87.25%** |
| **Precision** | 87.27% |
| **Recall** | 84.81% |
| **F1 Score** | 86.02% |

## Features

- 🖼️ **Image Analysis** — Upload an image to detect deepfakes with Grad-CAM heatmap visualization
- 🎬 **Video Analysis** — Upload a video for frame-by-frame deepfake detection
- 📸 **Webcam Capture** — Real-time deepfake detection from webcam
- 🔬 **Explainability** — Grad-CAM overlays show what the model focuses on

## Architecture

- **Backbone**: ResNet-50 (ImageNet pre-trained, progressively fine-tuned)
- **Classifier**: Dropout → FC(2048→512) → ReLU → BN → FC(512→1)
- **Face Detection**: OpenCV DNN (SSD-based, real-time)
- **Training**: Label smoothing, focal loss, cosine annealing, progressive unfreezing

## Dataset

Trained on the [Ciplab Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) dataset (~4,000 images).

## Usage

Upload any image containing a face, and the model will classify it as **REAL** or **FAKE** with a confidence score.

## Limitations

- Trained primarily on face-swap style deepfakes
- Requires a clearly visible face in the image
- CPU inference only (optimized for broad accessibility)
