"""FastAPI application for the Deepfake Detection API.

Endpoints
---------
- ``GET  /health``         — Service health check.
- ``GET  /model/info``     — Model architecture & parameter details.
- ``POST /predict/image``  — Upload an image for deepfake detection.
- ``POST /predict/video``  — Upload a video for deepfake detection.
"""

import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.inference import DeepfakePredictor
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    VideoResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(_PROJECT_ROOT, "models", "best_model.pth"),
)
DEVICE = "cpu"

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/x-msvideo",
    "video/quicktime",
}

# ---------------------------------------------------------------------------
# Application lifespan — loads the predictor once at startup
# ---------------------------------------------------------------------------
predictor: Optional[DeepfakePredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; release resources on shutdown."""
    global predictor  # noqa: PLW0603
    logger.info("Loading DeepfakePredictor (model_path=%s, device=%s) …", MODEL_PATH, DEVICE)
    predictor = DeepfakePredictor(model_path=MODEL_PATH, device=DEVICE)
    logger.info(
        "Predictor ready — model_loaded=%s, device=%s",
        predictor.is_loaded,
        predictor.device,
    )
    yield
    # Shutdown — nothing to clean up explicitly
    logger.info("Shutting down Deepfake Detection API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deepfake Detection API",
    description=(
        "A production-grade REST API for detecting deepfake images and "
        "videos using a fine-tuned ResNet-50 model.  All inference runs "
        "on CPU for broad hardware compatibility."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development / HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
#  Endpoints
# ===================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health() -> HealthResponse:
    """Return the current health status of the service."""
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False, device=DEVICE)
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        device=str(predictor.device),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Model information",
)
async def model_info() -> ModelInfoResponse:
    """Return architecture and parameter details for the loaded model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    model = predictor.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return ModelInfoResponse(
        architecture="ResNet50",
        parameters={
            "total_params": total_params,
            "trainable_params": trainable_params,
            "dropout": 0.5,
            "num_classes": 1,
            "pretrained_backbone": True,
        },
        input_size=224,
        device=str(predictor.device),
    )


@app.post(
    "/predict/image",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict deepfake from an image",
)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    """Upload a single image (JPEG / PNG / WebP) for deepfake detection.

    Returns 422 if no face is detected in the image.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}"
            ),
        )

    # Read file bytes
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Run prediction
    result = predictor.predict_image(image_bytes)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the uploaded image. Please upload an image containing a clearly visible face.",
        )

    return PredictionResponse(**result)


@app.post(
    "/predict/video",
    response_model=VideoResponse,
    tags=["Prediction"],
    summary="Predict deepfake from a video",
)
async def predict_video(
    file: UploadFile = File(...),
    num_frames: int = Query(
        default=15,
        ge=1,
        le=120,
        description="Number of frames to uniformly sample from the video.",
    ),
) -> VideoResponse:
    """Upload a video (MP4) for deepfake detection.

    The service uniformly samples ``num_frames`` frames from the video,
    analyses each one, and returns an aggregated prediction.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate content type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_VIDEO_TYPES))}"
            ),
        )

    # Save to a temp file so OpenCV can seek through it
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
            tmp.write(content)

        result = predictor.predict_video(tmp_path, num_frames=num_frames)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error during video prediction.")
        raise HTTPException(status_code=500, detail=f"Video prediction failed: {exc}")
    finally:
        # Clean up the temp file
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Could not delete temp file %s", tmp_path)

    return VideoResponse(**result)


# ---------------------------------------------------------------------------
# Entrypoint (for quick local testing: ``python api/main.py``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
