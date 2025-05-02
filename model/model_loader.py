import pandas as pd
import os
import torch
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


import torch.nn as nn
import torchvision.models as models

class ResNet18Binary(nn.Module):
    def __init__(self):
        super(ResNet18Binary, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # Modify first conv layer to accept grayscale
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change final FC layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import torch
import os
import copy

def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > 0.5).int().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs),
    }

def train_model(model, train_loader, val_loader, epochs=10, patience=3,
                checkpoint_path="/content/drive/MyDrive/lvh_resume_checkpoint.pt"):

    best_auc = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    start_epoch = 0

    # Resume from checkpoint if available
    if os.path.exists(checkpoint_path):
        print("📦 Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_auc = checkpoint["best_auc"]
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        best_model_wts = checkpoint["best_model_wts"]

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{epochs}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader)
        val_auc = val_metrics["auc"]

        tqdm.write(f"\n📊 Epoch {epoch+1}: "
                   f"Train Loss={avg_train_loss:.4f} | "
                   f"Val AUC={val_auc:.4f} | "
                   f"F1={val_metrics['f1']:.4f} | "
                   f"Acc={val_metrics['accuracy']:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            tqdm.write("✅ New best model found!")
        else:
            epochs_no_improve += 1
            tqdm.write("⚠️ No improvement.")

        # Save full checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "epochs_no_improve": epochs_no_improve,
            "best_model_wts": best_model_wts
        }, checkpoint_path)

        tqdm.write(f"💾 Checkpoint saved after Epoch {epoch+1}")

        if epochs_no_improve >= patience:
            tqdm.write("⏹️ Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    return model


# Load best model from checkpoint
checkpoint = torch.load("/content/drive/MyDrive/lvh_resume_checkpoint.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["best_model_wts"])

# Evaluate on test set
test_metrics = evaluate(model, test_loader)
print("📊 Test Set Metrics:", test_metrics)


# Load from full checkpoint first
checkpoint = torch.load("/content/drive/MyDrive/lvh_resume_checkpoint.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["best_model_wts"])

# Save just the model weights
torch.save(model.state_dict(), "/content/drive/MyDrive/lvh_best_model.pt")
print("✅ Best model saved as lvh_best_model.pt in Google Drive")


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def threshold_tuner(model, loader, thresholds=np.arange(0.1, 0.91, 0.05)):
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_f1 = 0
    best_acc = 0
    best_thresh_f1 = 0
    best_thresh_acc = 0

    print(f"{'Thresh':<8}{'Accuracy':<10}{'F1':<10}{'Precision':<10}{'Recall':<10}")
    print("-" * 50)

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)

        print(f"{thresh:<8.2f}{acc:<10.4f}{f1:<10.4f}{prec:<10.4f}{rec:<10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = thresh

        if acc > best_acc:
            best_acc = acc
            best_thresh_acc = thresh

    print("\n✅ Best F1 Threshold:", best_thresh_f1, f"→ F1: {best_f1:.4f}")
    print("✅ Best Accuracy Threshold:", best_thresh_acc, f"→ Accuracy: {best_acc:.4f}")


checkpoint = torch.load("/content/drive/MyDrive/lvh_resume_checkpoint.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["best_model_wts"])


import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# === Load best model from checkpoint ===
checkpoint = torch.load("/content/drive/MyDrive/lvh_resume_checkpoint.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["best_model_wts"])
model.eval()

# === Evaluation Parameters ===
threshold = 0.90  # You can change this to 0.6, 0.5, etc.

# === Evaluate on Test Set ===
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
preds = (all_probs >= threshold).astype(int)

# === Metrics ===
acc = accuracy_score(all_labels, preds)
prec = precision_score(all_labels, preds, zero_division=0)
rec = recall_score(all_labels, preds, zero_division=0)
f1 = f1_score(all_labels, preds, zero_division=0)
auc = roc_auc_score(all_labels, all_probs)

# === Print Results ===
print(f"📊 Evaluation at Threshold = {threshold:.2f}")
print(f"----------------------------------------")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC AUC      : {auc:.4f}")


# Load the best weights from checkpoint
checkpoint = torch.load("/content/drive/MyDrive/lvh_resume_checkpoint.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["best_model_wts"])

# Save as F1-balanced version
torch.save(model.state_dict(), "/content/drive/MyDrive/lvh_model_f1_balanced.pt")
print("✅ Saved F1-balanced model as lvh_model_f1_balanced.pt")


import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F

def generate_gradcam(model, image_tensor, target_class, final_conv_layer):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks
    handle_fw = final_conv_layer.register_forward_hook(forward_hook)
    handle_bw = final_conv_layer.register_backward_hook(backward_hook)

    # Forward pass
    image_tensor = image_tensor.unsqueeze(0).to(device)
    output = model(image_tensor)
    pred_prob = torch.sigmoid(output).item()
    pred_label = int(pred_prob > 0.5)

    # Backward pass for target class
    model.zero_grad()
    loss = F.binary_cross_entropy_with_logits(output.squeeze(1), torch.tensor([float(target_class)]).to(device))
    loss.backward()

    # Get gradients and activations
    grads = gradients[0].squeeze(0).cpu().detach().numpy()
    acts = activations[0].squeeze(0).cpu().detach().numpy()

    # Calculate Grad-CAM
    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    # Clean up hooks
    handle_fw.remove()
    handle_bw.remove()

    return cam, pred_prob, pred_label

def show_gradcam(model, dataset, idx, target_class):
    img_tensor, label = dataset[idx]
    cam, prob, pred = generate_gradcam(model, img_tensor, target_class, model.model.layer4[-1])

    # Prepare base image
    img = img_tensor.squeeze(0).cpu().numpy()
    img = (img * 0.5 + 0.5) * 255  # Unnormalize
    img = img.astype(np.uint8)

    # Convert to RGB for overlay
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_color, 0.5, heatmap, 0.5, 0)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"GT: {int(label)}, Pred: {pred} (Prob: {prob:.2f})")
    plt.axis('off')
    plt.show()

# Example usage:
# Show Grad-CAM for test sample index 3 (change as needed)
show_gradcam(model, test_dataset, idx=3, target_class=1)


import os
import shutil

# Setup output folder
output_dir = "/content/gradcam_outputs"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

def generate_topN_gradcams(model, dataset, loader, top_n=5, target_class=1):
    print(f"🔍 Generating Grad-CAMs for Top {top_n} high-confidence class-{target_class} predictions...")

    model.eval()
    probs = []
    indices = []

    # Step 1: Collect all predictions and their probabilities
    with torch.no_grad():
        for i, (img, label) in enumerate(dataset):
            img_tensor = img.unsqueeze(0).to(device)
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()

            if int(prob > 0.5) == target_class:
                probs.append(prob)
                indices.append(i)

    # Step 2: Sort by confidence
    top_idxs = [x for _, x in sorted(zip(probs, indices), reverse=True)][:top_n]

    # Step 3: Generate Grad-CAMs
    for idx in top_idxs:
        img_tensor, label = dataset[idx]
        cam, prob, pred = generate_gradcam(model, img_tensor, target_class, model.model.layer4[-1])

        # Prepare base image
        img = img_tensor.squeeze(0).cpu().numpy()
        img = (img * 0.5 + 0.5) * 255  # Unnormalize
        img = img.astype(np.uint8)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_color, 0.5, heatmap, 0.5, 0)

        filename = f"{output_dir}/gradcam_idx{idx}_prob{prob:.2f}_pred{pred}_gt{int(label)}.png"
        cv2.imwrite(filename, overlay)
        print(f"✅ Saved: {filename}")

# 🔁 Call the function
generate_topN_gradcams(model, test_dataset, test_loader, top_n=5, target_class=1)


import pandas as pd

def save_test_predictions_csv(model, loader, dataset, threshold=0.90, save_path="/content/test_predictions.csv"):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend((probs >= threshold).astype(int))

    # Build dataframe
    df = pd.DataFrame({
        "ground_truth": all_labels,
        "predicted_prob": all_probs,
        "predicted_label": all_preds
    })

    df.to_csv(save_path, index=False)
    print(f"✅ CSV saved at: {save_path}")

# Run it:
save_test_predictions_csv(model, test_loader, test_dataset, threshold=0.90)


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNet18Binary(nn.Module):
    def __init__(self):
        super(ResNet18Binary, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)


# Load the best model
model = ResNet18Binary().to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/lvh_model_f1_balanced.pt", map_location=device))
model.eval()
