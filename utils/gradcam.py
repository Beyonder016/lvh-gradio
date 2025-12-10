# STEP 6: Grad-CAM function
def generate_gradcam(input_tensor, model, target_layer):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Hooks
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    prob = torch.sigmoid(output).item()
    pred_label = int(prob > 0.5)

    model.zero_grad()
    target = torch.tensor([[1.0]]).to(device) if pred_label == 1 else torch.tensor([[0.0]]).to(device)
    output.backward(gradient=target)

    grads = gradients[0]
    acts = activations[0]
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]
    heatmap = torch.mean(acts, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-6)

    fh.remove()
    bh.remove()

    return heatmap, prob, pred_label


# STEP 7: Inference and overlay
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_and_gradcam(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    heatmap, prob, pred = generate_gradcam(img_tensor, model, final_conv_layer)

    pred_class = "LVH" if pred == 1 else "No LVH"
    label = f"{pred_class} (Confidence: {prob:.2f})"

    # Overlay Grad-CAM
    base = np.array(img.resize((224, 224)).convert("RGB"))
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.5, cam, 0.5, 0)

    return Image.fromarray(overlay), label


# STEP 8: Launch Gradio demo
interface = gr.Interface(
    fn=predict_and_gradcam,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray (.JPG or PNG)"),
    outputs=[
        gr.Image(type="pil", label="Grad-CAM Heatmap"),
        gr.Text(label="Prediction")
    ],
    title="LVH Detection Demo",
    description="Upload a chest X-ray to see LVH prediction and model attention"
)

interface.launch(debug=True, share=True)
