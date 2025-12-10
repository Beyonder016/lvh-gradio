"""Streamlit app for LVH detection with Grad-CAM visualization."""
from pathlib import Path
from typing import Tuple

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models, transforms

# -------------------------------------------------------------------
# Streamlit basic page config MUST be the first Streamlit command
# -------------------------------------------------------------------
st.set_page_config(page_title="LVH Detection via Chest X-Ray", layout="wide")

# -------------------------------------------------------------------
# Glassmorphism CSS (simple + safe)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        /* Force Streamlit components to render in light mode
           so our custom dark theme is visible & readable */
        color-scheme: light;
    }

    /* === Dark background (no blobs, simple & safe) === */
    .stApp {
        background: linear-gradient(135deg, #05070b 0%, #0c1117 40%, #05070b 100%);
    }

    /* === Main frosted dark glass panel === */
    .main .block-container {
        max-width: 1150px;
        margin: 2.5rem auto 3.5rem auto;
        padding: 2.8rem 3rem;

        /* dark glass */
        background: rgba(20, 24, 35, 0.78);
        border-radius: 26px;
        border: 1px solid rgba(255, 255, 255, 0.06);

        backdrop-filter: blur(26px) saturate(140%);
        -webkit-backdrop-filter: blur(26px) saturate(140%);

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.65),
            inset 0 0 40px rgba(255, 255, 255, 0.03);

        color: #f4f4f5;
    }

    /* === Headings & text === */
    h1, h2, h3, h4 {
        color: #f9fafb !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }

    p, label, span {
        color: #e5e7eb !important;
    }

    /* === File uploader: dark glass chip === */
    .stFileUploader {
        background: rgba(17, 24, 39, 0.8) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border-radius: 18px !important;
        padding: 1.1rem !important;
        border: 1px solid rgba(148, 163, 184, 0.6);
    }

    /* === Buttons: blue accent === */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #f9fafb;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        box-shadow: 0 10px 26px rgba(37, 99, 235, 0.5);
    }

    .stButton > button:hover {
        filter: brightness(1.1);
        transform: translateY(-2px);
        transition: 0.15s ease-in-out;
    }

    /* === Prediction pill === */
    .prediction-pill {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.7);
        font-size: 0.9rem;
        color: #f9fafb;
        margin-top: 0.25rem;
    }

    /* === Images === */
    img {
        border-radius: 16px !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7);
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# --- Model utilities -------------------------------------------------------
REPO_ID = "Beyonder016/lvh-detector"
WEIGHTS_FILENAME = "model.pt"
LOCAL_WEIGHTS = Path("lvh_model_f1_balanced.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model() -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model.to(device)


def load_weights(model: torch.nn.Module, weight_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    cleaned_state = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()
    return model


@st.cache_resource
def load_model() -> torch.nn.Module:
    model = build_model()
    load_errors: list[str] = []

    if LOCAL_WEIGHTS.exists():
        try:
            return load_weights(model, LOCAL_WEIGHTS)
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"Local weights failed: {exc}")
            st.warning("Local weights could not be loaded, falling back to Hugging Face Hub.")

    try:
        hub_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILENAME)
        return load_weights(model, Path(hub_path))
    except Exception as exc:  # noqa: BLE001
        load_errors.append(str(exc))
        joined = "\n".join(load_errors)
        raise RuntimeError(f"Unable to load model weights. Details:\n{joined}") from exc


transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return transform(image.convert("RGB")).unsqueeze(0).to(device)


def get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise RuntimeError("No convolutional layer found for Grad-CAM.")


def generate_gradcam(model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[float, np.ndarray]:
    model.eval()
    input_tensor = input_tensor.clone().requires_grad_(True)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    target_layer = get_last_conv_layer(model)

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, __, grad_output):
        gradients["value"] = grad_output[0]

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_backward_hook(backward_hook)

    logits = model(input_tensor)
    prob = torch.sigmoid(logits).squeeze()

    model.zero_grad()
    logits.backward(torch.ones_like(logits))

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1)).squeeze().detach()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    fwd_handle.remove()
    bwd_handle.remove()

    return prob.item(), cam.cpu().numpy()


def overlay_heatmap(original: Image.Image, heatmap: np.ndarray) -> Image.Image:
    heatmap_img = Image.fromarray(np.uint8(cm.jet(heatmap)[:, :, :3] * 255)).resize(original.size)
    return Image.blend(original.convert("RGB"), heatmap_img, alpha=0.5)


# --- Streamlit app content -------------------------------------------------

st.title("💓 LVH Detection from Chest X-Ray")
st.markdown(
    "Upload a chest X-ray image to detect **Left Ventricular Hypertrophy (LVH)** "
    "and visualize where the model is focusing using **Grad-CAM**."
)

model = load_model()

st.markdown("### 1️⃣ Upload Chest X-ray")
uploaded = st.file_uploader("Drag & drop or browse a PNG/JPG image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)

    # Side-by-side layout
    st.markdown("### 2️⃣ Model Prediction & Grad-CAM")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Input X-ray")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        gradcam_placeholder = st.empty()
        info_placeholder = st.empty()

    # Prediction
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        probability = torch.sigmoid(model(input_tensor)).item()

    label = "LVH" if probability >= 0.5 else "No LVH"

    st.markdown(
        f"""
        <h3>3️⃣ Prediction</h3>
        <div class="prediction-pill">
            Predicted label: <b>{label}</b> &nbsp;|&nbsp;
            LVH confidence: <b>{probability:.2%}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    generate_btn = st.button("🔥 Generate Grad-CAM Heatmap")

    if generate_btn:
        cam_prob, heatmap = generate_gradcam(model, input_tensor)
        overlay = overlay_heatmap(image, heatmap)

        with col2:
            gradcam_placeholder.image(
                overlay,
                caption=f"Grad-CAM Heatmap (LVH confidence: {cam_prob:.2%})",
                use_container_width=True,
            )
            info_placeholder.info(
                "Red/yellow regions indicate areas the model considers most important for its LVH decision."
            )
    else:
        with col2:
            info_placeholder.markdown(
                "_Click **Generate Grad-CAM Heatmap** to visualize model attention._"
            )
else:
    st.info("Upload a chest X-ray image to start the analysis.")
