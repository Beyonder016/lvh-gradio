import gradio as gr
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# Load model from Hugging Face Hub
model_path = hf_hub_download(repo_id="Beyonder016/lvh-detector", filename="model.pt")
model = torch.load(model_path, map_location="cpu")
model.eval()

# Grad-CAM utility
def get_last_conv_layer(model):
    for name, module in reversed(model._modules.items()):
        if isinstance(module, torch.nn.Conv2d):
            return name, module
        if hasattr(module, 'children'):
            result = get_last_conv_layer(module)
            if result: return result
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

    output = model(img_tensor)
    pred_class = output.argmax().item()
    confidence = F.softmax(output, dim=1)[0][pred_class].item()

    model.zero_grad()
    output[0][pred_class].backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().numpy(), 0)
    heatmap /= np.max(heatmap)

    heatmap = Image.fromarray(np.uint8(plt.cm.jet(heatmap)[:, :, :3]*255))
    heatmap = heatmap.resize((img.width, img.height))
    overlay = Image.blend(img.convert("RGB"), heatmap, alpha=0.5)

    fwd_handle.remove()
    bwd_handle.remove()

    label = "LVH" if pred_class == 1 else "No LVH"
    return overlay, f"{label} (Confidence: {confidence:.2f})"

interface = gr.Interface(
    fn=predict_with_gradcam,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray (.JPG or PNG)"),
    outputs=[
        gr.Image(label="Grad-CAM Heatmap"),
        gr.Textbox(label="Prediction")
    ],
    title="LVH Detection Demo",
    description="Upload a chest X-ray to see LVH prediction and model attention"
)

if __name__ == "__main__":
    interface.launch()