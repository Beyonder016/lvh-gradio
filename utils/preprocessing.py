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


!pip install torchvision --quiet

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


from torchvision import transforms

# Augmentations for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# No augmentations for val/test
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# Define test transform (if not already defined)
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Recreate dataset and loader
test_dataset = LVHDataset(df_test, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print("✅ Test loader ready!")


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


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_and_gradcam(img):
    image = transform(img).unsqueeze(0).to(device)

    # Get prediction and heatmap
    heatmap, prob = generate_gradcam(image, model, model.model.layer4[-1])
    pred_class = "LVH" if prob >= 0.9 else "No LVH"
    prob_text = f"{prob:.4f} → {pred_class}"

    # Convert heatmap to image
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    img_cv = np.array(img.resize((224, 224)).convert("RGB"))
    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_img, 0.4, 0)

    return Image.fromarray(superimposed_img), prob_text
