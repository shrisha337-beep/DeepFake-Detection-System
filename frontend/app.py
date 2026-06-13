"""Gradio frontend for the Deepfake Detection System.

This is the main entry-point for Hugging Face Spaces.  It provides:

- **Image Tab** — Upload an image → deepfake prediction with confidence
  bar and optional Grad-CAM heatmap overlay.
- **Video Tab** — Upload a video → aggregated prediction with per-frame
  analysis chart.
- **Webcam Tab** — Live webcam capture with real-time detection.
- **About Tab** — Model architecture, how it works, and limitations.

Launch
------
    python frontend/app.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Use non-interactive backend for Matplotlib (server-side rendering)
matplotlib.use("Agg")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Lazy predictor & Grad-CAM initialisation
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(_PROJECT_ROOT, "models", "best_model.pth"),
)

predictor = None
gradcam_fn = None


def _get_predictor():
    """Lazily initialise the predictor (import-time failures are OK)."""
    global predictor  # noqa: PLW0603
    if predictor is not None:
        return predictor
    try:
        from api.inference import DeepfakePredictor

        predictor = DeepfakePredictor(model_path=MODEL_PATH, device="cpu")
        logger.info("DeepfakePredictor initialised (loaded=%s).", predictor.is_loaded)
    except Exception as exc:
        logger.warning("Could not initialise DeepfakePredictor: %s", exc)
        predictor = None
    return predictor


def _get_gradcam():
    """Lazily import the Grad-CAM helper."""
    global gradcam_fn  # noqa: PLW0603
    if gradcam_fn is not None:
        return gradcam_fn
    try:
        from src.evaluation.gradcam import generate_gradcam

        gradcam_fn = generate_gradcam
        logger.info("Grad-CAM function loaded.")
    except Exception as exc:
        logger.warning("Grad-CAM unavailable: %s", exc)
        gradcam_fn = None
    return gradcam_fn


# ===================================================================
#  Helper utilities
# ===================================================================

def _confidence_html(prediction: str, confidence: float) -> str:
    """Build a styled HTML card showing the prediction result."""
    is_fake = prediction == "FAKE"
    emoji = "🚨" if is_fake else "✅"
    colour = "#ef4444" if is_fake else "#22c55e"
    bg = "rgba(239,68,68,0.10)" if is_fake else "rgba(34,197,94,0.10)"
    bar_pct = int(confidence * 100)

    return f"""
    <div style="
        background:{bg};
        border:2px solid {colour};
        border-radius:16px;
        padding:24px 28px;
        text-align:center;
        font-family:Inter,system-ui,sans-serif;
    ">
        <div style="font-size:48px;margin-bottom:8px;">{emoji}</div>
        <div style="font-size:28px;font-weight:800;color:{colour};margin-bottom:4px;">
            {prediction}
        </div>
        <div style="font-size:15px;color:#94a3b8;margin-bottom:14px;">
            Confidence: <strong style="color:{colour};">{bar_pct}%</strong>
        </div>
        <div style="
            background:#1e293b;
            border-radius:8px;
            height:14px;
            overflow:hidden;
        ">
            <div style="
                width:{bar_pct}%;
                height:100%;
                background:linear-gradient(90deg,{colour},{colour}cc);
                border-radius:8px;
                transition:width .4s ease;
            "></div>
        </div>
    </div>
    """


def _no_face_html() -> str:
    """Return HTML for a 'no face detected' message."""
    return """
    <div style="
        background:rgba(234,179,8,0.10);
        border:2px solid #eab308;
        border-radius:16px;
        padding:24px 28px;
        text-align:center;
        font-family:Inter,system-ui,sans-serif;
    ">
        <div style="font-size:48px;margin-bottom:8px;">⚠️</div>
        <div style="font-size:22px;font-weight:700;color:#eab308;">
            No Face Detected
        </div>
        <div style="font-size:14px;color:#94a3b8;margin-top:6px;">
            Please upload an image containing a clearly visible face.
        </div>
    </div>
    """


def _error_html(msg: str) -> str:
    """Return HTML for a generic error message."""
    return f"""
    <div style="
        background:rgba(239,68,68,0.08);
        border:2px solid #ef4444;
        border-radius:16px;
        padding:20px 24px;
        text-align:center;
        font-family:Inter,system-ui,sans-serif;
    ">
        <div style="font-size:36px;margin-bottom:8px;">❌</div>
        <div style="font-size:16px;color:#f87171;">{msg}</div>
    </div>
    """


def _frame_chart(frame_predictions: list[dict]) -> Optional[plt.Figure]:
    """Build a per-frame fake-probability chart."""
    if not frame_predictions:
        return None

    indices = [fp["frame_index"] for fp in frame_predictions]
    probs = [fp["fake_probability"] for fp in frame_predictions]

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Colour each bar by probability
    colours = ["#ef4444" if p >= 0.5 else "#22c55e" for p in probs]
    ax.bar(range(len(indices)), probs, color=colours, edgecolor="none", width=0.7)
    ax.axhline(y=0.5, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("Frame Index", color="#e2e8f0", fontsize=10)
    ax.set_ylabel("Fake Probability", color="#e2e8f0", fontsize=10)
    ax.set_title("Per-Frame Analysis", color="#e2e8f0", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([str(i) for i in indices], fontsize=7, color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")

    fig.tight_layout()
    return fig


# ===================================================================
#  Prediction callbacks
# ===================================================================

def predict_image_cb(image: Optional[np.ndarray]):
    """Gradio callback for the Image tab."""
    if image is None:
        return _error_html("No image provided."), None

    pred = _get_predictor()
    if pred is None:
        return _error_html("Model not available. Check server logs."), None

    result = pred.predict_numpy(image)
    if result is None:
        return _no_face_html(), None

    # Attempt Grad-CAM overlay
    gradcam_image = None
    gc = _get_gradcam()
    if gc is not None and pred.is_loaded:
        try:
            # Extract face for Grad-CAM (same pipeline as prediction)
            face = pred.face_extractor.extract(image)
            if face is not None:
                # Create model input tensor
                transformed = pred.transforms(image=face)
                input_tensor = transformed["image"].unsqueeze(0)
                # Generate heatmap overlay
                gradcam_image, _ = gc(
                    model=pred.model,
                    input_tensor=input_tensor,
                    original_image_np=face,
                )
        except Exception as exc:
            logger.warning("Grad-CAM generation failed: %s", exc)

    html = _confidence_html(result["prediction"], result["confidence"])
    return html, gradcam_image


def predict_video_cb(video_path: Optional[str], num_frames: int = 15):
    """Gradio callback for the Video tab."""
    if video_path is None:
        return _error_html("No video provided."), None

    pred = _get_predictor()
    if pred is None:
        return _error_html("Model not available. Check server logs."), None

    result = pred.predict_video(video_path, num_frames=int(num_frames))

    if result["num_frames_analyzed"] == 0:
        return _no_face_html(), None

    html = _confidence_html(result["prediction"], result["confidence"])
    # Append summary line
    html += f"""
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:13px;
        margin-top:12px;
        font-family:Inter,system-ui,sans-serif;
    ">
        Analysed <strong>{result['num_frames_analyzed']}</strong> frames
        · Avg fake probability: <strong>{result['avg_fake_probability']:.2%}</strong>
    </div>
    """

    chart = _frame_chart(result.get("frame_predictions", []))
    return html, chart


def predict_webcam_cb(image: Optional[np.ndarray]):
    """Gradio callback for the Webcam tab (single-shot capture)."""
    if image is None:
        return _error_html("No capture received.")

    pred = _get_predictor()
    if pred is None:
        return _error_html("Model not available.")

    result = pred.predict_numpy(image)
    if result is None:
        return _no_face_html()

    return _confidence_html(result["prediction"], result["confidence"])


# ===================================================================
#  Custom CSS
# ===================================================================

CUSTOM_CSS = """
/* ── Global polish ──────────────────────────────── */
.gradio-container {
    max-width: 960px !important;
    margin: auto !important;
}

/* ── Gradient header ────────────────────────────── */
#header-row {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1a1a2e 100%);
    border-radius: 16px;
    padding: 32px 28px 24px;
    margin-bottom: 8px;
    text-align: center;
    border: 1px solid #334155;
}
#header-row h1 {
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 900;
    margin: 0 0 6px;
}
#header-row p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
}

/* ── Tab styling ────────────────────────────────── */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ── Cards ──────────────────────────────────────── */
.result-card {
    border-radius: 16px;
    overflow: hidden;
}

/* ── Footer ─────────────────────────────────────── */
#footer-row {
    text-align: center;
    padding: 16px 0 4px;
    color: #475569;
    font-size: 0.8rem;
}
"""


# ===================================================================
#  About tab content (Markdown)
# ===================================================================

ABOUT_MD = """
## 🔬 How It Works

This system uses a **fine-tuned ResNet-50** convolutional neural network
to distinguish real face images from AI-generated deepfakes.

### Pipeline

1. **Face Extraction** — An OpenCV DNN face detector locates and crops
   the primary face region from the input image.
2. **Preprocessing** — The face crop is resized to 224 × 224 and
   normalised using ImageNet statistics.
3. **Classification** — The ResNet-50 backbone produces a single logit
   which is passed through a sigmoid to obtain a *fake probability*.
4. **Thresholding** — A probability ≥ 0.5 → **FAKE**; otherwise **REAL**.

### Architecture Details

| Component | Details |
|-----------|---------|
| **Backbone** | ResNet-50 (ImageNet pre-trained) |
| **Classifier** | Dropout (0.5) → Linear (2048 → 1) |
| **Input size** | 224 × 224 × 3 |
| **Output** | Sigmoid probability [0, 1] |
| **Device** | CPU |

### Grad-CAM Explainability

The **Image** tab overlays a Grad-CAM heatmap generated from the last
convolutional layer (`layer4`).  Red regions indicate areas the model
focuses on most, providing interpretable evidence for its decision.

### Limitations

- Trained primarily on face-swap style deepfakes; may underperform on
  other manipulation types (e.g., lip-sync, full reenactment).
- Requires a clearly visible face — profile views or heavy occlusion
  may cause the face detector to miss the subject.
- CPU-only inference means throughput is limited for high-volume
  production use; consider GPU deployment for scale.
- Not designed for adversarial robustness — intentionally crafted
  adversarial perturbations may fool the classifier.

---

<div style="text-align:center;color:#64748b;font-size:0.85rem;margin-top:12px;">
Built with ❤️ using PyTorch · FastAPI · Gradio
</div>
"""


# ===================================================================
#  Build the Gradio Blocks UI
# ===================================================================

def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f172a",
        body_background_fill_dark="#0f172a",
        block_background_fill="#1e293b",
        block_background_fill_dark="#1e293b",
        block_border_color="#334155",
        block_border_color_dark="#334155",
        block_label_text_color="#e2e8f0",
        block_label_text_color_dark="#e2e8f0",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        input_background_fill="#0f172a",
        input_background_fill_dark="#0f172a",
        button_primary_background_fill="linear-gradient(135deg, #3b82f6, #8b5cf6)",
        button_primary_background_fill_dark="linear-gradient(135deg, #3b82f6, #8b5cf6)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
    )

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="Deepfake Detector",
        analytics_enabled=False,
    ) as demo:

        # ── Header ───────────────────────────────────────────────
        with gr.Row(elem_id="header-row"):
            gr.HTML(
                """
                <h1>🛡️ Deepfake Detector</h1>
                <p>Upload an image or video to detect AI-generated deepfakes
                using a fine-tuned ResNet-50 model.</p>
                """
            )

        # ── Tabs ─────────────────────────────────────────────────
        with gr.Tabs():

            # ═══════════════════ IMAGE TAB ═══════════════════════
            with gr.TabItem("🖼️ Image", id="image-tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            sources=["upload", "clipboard"],
                            height=380,
                        )
                        img_btn = gr.Button(
                            "🔍 Analyse Image",
                            variant="primary",
                            size="lg",
                        )
                        gr.Examples(
                            examples=[
                                ["examples/real_1.jpg"],
                                ["examples/fake_1.jpg"],
                            ],
                            inputs=[img_input],
                            label="Try an example",
                            cache_examples=False,
                        )
                    with gr.Column(scale=1):
                        img_result = gr.HTML(
                            label="Prediction",
                            elem_classes=["result-card"],
                        )
                        img_gradcam = gr.Image(
                            label="Grad-CAM Heatmap",
                            type="numpy",
                            interactive=False,
                            height=300,
                        )

                img_btn.click(
                    fn=predict_image_cb,
                    inputs=[img_input],
                    outputs=[img_result, img_gradcam],
                )

            # ═══════════════════ VIDEO TAB ═══════════════════════
            with gr.TabItem("🎬 Video", id="video-tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                            height=380,
                        )
                        vid_frames = gr.Slider(
                            label="Frames to sample",
                            minimum=5,
                            maximum=60,
                            step=1,
                            value=15,
                        )
                        vid_btn = gr.Button(
                            "🔍 Analyse Video",
                            variant="primary",
                            size="lg",
                        )
                    with gr.Column(scale=1):
                        vid_result = gr.HTML(
                            label="Prediction",
                            elem_classes=["result-card"],
                        )
                        vid_chart = gr.Plot(
                            label="Per-Frame Analysis",
                        )

                vid_btn.click(
                    fn=predict_video_cb,
                    inputs=[vid_input, vid_frames],
                    outputs=[vid_result, vid_chart],
                )

            # ═══════════════════ WEBCAM TAB ══════════════════════
            with gr.TabItem("📸 Webcam", id="webcam-tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam_input = gr.Image(
                            label="Webcam Capture",
                            type="numpy",
                            sources=["webcam"],
                            height=380,
                        )
                        webcam_btn = gr.Button(
                            "🔍 Analyse Capture",
                            variant="primary",
                            size="lg",
                        )
                    with gr.Column(scale=1):
                        webcam_result = gr.HTML(
                            label="Prediction",
                            elem_classes=["result-card"],
                        )

                webcam_btn.click(
                    fn=predict_webcam_cb,
                    inputs=[webcam_input],
                    outputs=[webcam_result],
                )

            # ═══════════════════ ABOUT TAB ═══════════════════════
            with gr.TabItem("ℹ️ About", id="about-tab"):
                gr.Markdown(ABOUT_MD)

        # ── Footer ───────────────────────────────────────────────
        with gr.Row(elem_id="footer-row"):
            gr.HTML(
                "<p>Deepfake Detection System v1.0 · CPU Inference · "
                "ResNet-50 Backbone</p>"
            )

    return demo


# ===================================================================
#  Main
# ===================================================================

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
