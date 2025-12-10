# 🫀 LVH Detection from Chest X-Rays

This Streamlit app detects **Left Ventricular Hypertrophy (LVH)** on chest X-ray images and uses Grad-CAM to highlight the regions influencing the prediction. Upload an image to see the model's classification and attention map side by side.

---

## 🚀 Quickstart (Streamlit)

```bash
git clone https://github.com/Beyonder016/lvh-gradio.git
cd lvh-gradio
pip install -r requirements.txt
streamlit run app_streamlit.py
```

When the app starts, open the provided local URL to upload an image and view the prediction with its Grad-CAM heatmap.

---

## 📦 Pretrained Models

Download the latest pretrained weights from the **[Releases Page](https://github.com/Beyonder016/lvh-gradio/releases)**.

Available model file:

- `resnet18_balanced.pth` (place the downloaded `.pth` or `.pt` file at the project root or in the `model/` directory)

> The app will automatically load `lvh_model_f1_balanced.pt` if present locally, otherwise it falls back to downloading `model.pt` from the Hugging Face repository `Beyonder016/lvh-detector`.

---

## 🧪 Try with Sample Images

Use these folders for quick tests:

- 📂 [samples/LVH](samples/LVH) — X-rays with confirmed LVH
- 📂 [samples/No_LVH](samples/No_LVH) — X-rays without LVH

Download any image and upload it in the Streamlit interface to see predictions and Grad-CAM visualizations.

---

## ⚙️ How It Works

1. Upload a `.jpg` or `.png` chest X-ray
2. The model predicts: **LVH** or **No LVH**
3. A Grad-CAM heatmap highlights the region used for the prediction

---

## 🛠 Built With

- PyTorch & torchvision
- Grad-CAM (model explainability)
- Streamlit (interactive web UI)
- Hugging Face Hub (model weights hosting)

---

## 🙌 Author

Made with ❤️ by [@Beyonder016](https://github.com/Beyonder016)
