import gradio as gr
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import torchvision.models as models

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


def _load_checkpoint() -> torch.Tensor:
    local_path = _find_local_model()
    if local_path:
        return torch.load(local_path, map_location="cpu")

    try:
        remote_path = hf_hub_download(
            repo_id="Beyonder016/lvh-detector", filename="model.pt"
        )
    except Exception as download_error:
        raise RuntimeError(
            "No local model file found and downloading from Hugging Face Hub failed. "
            "Place a model checkpoint inside the `model/` directory (e.g. `model.pt` "
            "or `resnet18_best.pth`) to run the app without internet access."
        ) from download_error

    return torch.load(remote_path, map_location="cpu")


model = get_model()

checkpoint = _load_checkpoint()
if "model" in checkpoint:
    raw_state_dict = checkpoint["model"]
else:
    raw_state_dict = checkpoint

def get_last_conv_layer(model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, torch.nn.Conv2d):
            return name, module
        if hasattr(module, 'children'):
            result = get_last_conv_layer(module)
            if result:
                return result
    return None

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

def predict_with_gradcam(img):
    img_tensor = transform(img.convert("L")).unsqueeze(0)
    img_tensor.requires_grad = True

    def forward_hook(module, input, output):
        global activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        global gradients
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

interface = gr.Interface(
    fn=predict_with_gradcam,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.Image(label="Grad-CAM Heatmap"),
        gr.Textbox(label="Prediction"),
    ],
    title="LVH Detection Demo",
    description="Upload a chest X-ray to predict Left Ventricular Hypertrophy",
)

if __name__ == "__main__":
    interface.launch()
