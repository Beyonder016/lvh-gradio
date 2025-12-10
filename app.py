import io
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as T

def get_model():
    model = models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model

MODEL_CANDIDATES = [
    Path("model/model.pt"),
    Path("model/resnet18_best.pth"),
    Path("model/resnet18_balanced.pth"),
    Path("model/resnet18_91acc.pth"),
]


def _find_local_model() -> Optional[Path]:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    generic_candidates = sorted(Path("model").glob("*.pt")) + sorted(
        Path("model").glob("*.pth")
    )
    return generic_candidates[0] if generic_candidates else None


def _load_checkpoint_bytes(upload: Optional[bytes] = None) -> Optional[bytes]:
    """Return checkpoint bytes from upload or local file, if present."""

    if upload:
        return upload

    local_path = _find_local_model()
    if local_path:
        return local_path.read_bytes()

    return None


@st.cache_resource(show_spinner=False)
def _prepare_model(checkpoint_bytes: bytes) -> torch.nn.Module:
    """Instantiate and load the model using cached resources."""

    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    model = get_model()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_last_conv_layer(model: torch.nn.Module):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, torch.nn.Conv2d):
            return name, module
        if hasattr(module, 'children'):
            result = get_last_conv_layer(module)
            if result:
                return result
    return None

transform = T.Compose(
    [T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5], [0.5])]
)


def predict_with_gradcam(img: Image.Image, model: torch.nn.Module) -> Tuple[Image.Image, str]:
    img_tensor = transform(img.convert("L")).unsqueeze(0)
    img_tensor.requires_grad = True

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    _, last_conv = get_last_conv_layer(model)
    fwd_handle = last_conv.register_forward_hook(forward_hook)
    bwd_handle = last_conv.register_backward_hook(backward_hook)

    logits = model(img_tensor)
    prob = torch.sigmoid(logits).item()
    label = "LVH" if prob >= 0.5 else "No LVH"
    confidence = prob if label == "LVH" else 1 - prob

    model.zero_grad()
    logits.backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)
    heatmap /= np.max(heatmap)

    heatmap = Image.fromarray(np.uint8(plt.cm.jet(heatmap)[:, :, :3] * 255))
    heatmap = heatmap.resize((img.width, img.height))
    overlay = Image.blend(img.convert("RGB"), heatmap, alpha=0.5)

    fwd_handle.remove()
    bwd_handle.remove()

    return overlay, f"{label} (Confidence: {confidence:.2f})"


def main():
    st.set_page_config(
        page_title="LVH Detection Demo", page_icon="🫀", layout="centered"
    )
    st.title("LVH Detection Demo")
    st.write(
        "Upload a chest X-ray to predict Left Ventricular Hypertrophy with a Grad-CAM "
        "visual explanation."
    )

    st.sidebar.header("Model Checkpoint")
    st.sidebar.write(
        "Place a model checkpoint in the `model/` folder or upload one here to run "
        "predictions without any internet access."
    )

    uploaded_checkpoint = st.sidebar.file_uploader(
        "Upload a PyTorch checkpoint", type=["pt", "pth"], accept_multiple_files=False
    )
    checkpoint_bytes = _load_checkpoint_bytes(
        uploaded_checkpoint.read() if uploaded_checkpoint else None
    )

    if checkpoint_bytes is None:
        st.error(
            "No model checkpoint found. Add a `.pt` or `.pth` file to the `model/` "
            "directory or upload one from the sidebar."
        )
        return

    with st.spinner("Loading model..."):
        model = _prepare_model(checkpoint_bytes)

    uploaded_image = st.file_uploader(
        "Upload Chest X-ray", type=["png", "jpg", "jpeg", "bmp", "tiff"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Input X-ray", use_container_width=True)

        if st.button("Run Prediction"):
            with st.spinner("Analyzing image and generating Grad-CAM..."):
                overlay, prediction_text = predict_with_gradcam(image, model)

            st.subheader("Prediction")
            st.success(prediction_text)
            st.subheader("Grad-CAM Heatmap")
            st.image(overlay, caption="Model explanation", use_container_width=True)


if __name__ == "__main__":
    main()
