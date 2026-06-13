"""Pydantic schemas for the Deepfake Detection API.

Defines request/response models for image prediction, video prediction,
health checks, and model information endpoints.
"""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for single image deepfake prediction."""

    prediction: str = Field(
        ...,
        description="Classification result: 'REAL' or 'FAKE'",
        examples=["FAKE"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction (0.0 to 1.0)",
        examples=[0.92],
    )
    fake_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw probability that the image is fake (0.0 to 1.0)",
        examples=[0.92],
    )
    face_detected: bool = Field(
        ...,
        description="Whether a face was detected in the image",
        examples=[True],
    )


class FramePrediction(BaseModel):
    """Prediction result for a single video frame."""

    frame_index: int = Field(
        ...,
        ge=0,
        description="Zero-based index of the sampled frame",
        examples=[0],
    )
    fake_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that this frame is fake",
        examples=[0.87],
    )


class VideoResponse(BaseModel):
    """Response schema for video deepfake prediction."""

    prediction: str = Field(
        ...,
        description="Overall classification result: 'REAL' or 'FAKE'",
        examples=["FAKE"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the overall prediction",
        examples=[0.89],
    )
    avg_fake_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average fake probability across all analyzed frames",
        examples=[0.89],
    )
    num_frames_analyzed: int = Field(
        ...,
        ge=0,
        description="Number of frames that were successfully analyzed",
        examples=[15],
    )
    frame_predictions: list[FramePrediction] = Field(
        default_factory=list,
        description="Per-frame prediction results",
    )


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str = Field(
        ...,
        description="Service status: 'healthy' or 'degraded'",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready for inference",
        examples=[True],
    )
    device: str = Field(
        ...,
        description="Compute device in use (e.g., 'cpu', 'cuda')",
        examples=["cpu"],
    )


class ModelInfoResponse(BaseModel):
    """Response schema for the model information endpoint."""

    architecture: str = Field(
        ...,
        description="Model architecture name",
        examples=["ResNet50"],
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Model parameter counts and configuration details",
        examples=[{
            "total_params": 23_508_032,
            "trainable_params": 23_508_032,
            "dropout": 0.5,
            "num_classes": 1,
        }],
    )
    input_size: int = Field(
        ...,
        description="Expected input image size (height = width)",
        examples=[224],
    )
    device: str = Field(
        ...,
        description="Compute device in use",
        examples=["cpu"],
    )
