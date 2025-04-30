[🟢 Click here to try the Live Demo](https://huggingface.co/spaces/Beyonder016/lvh-gradio)

---

# 🫀 LVH Detection from Chest X-Rays

This web app detects **Left Ventricular Hypertrophy (LVH)** using a deep learning model trained on chest X-ray images. It uses Grad-CAM to visualize the areas influencing the prediction.

---

## 🧪 Try with Sample Images

To quickly test the model, use these folders:

- 📂 [samples/LVH](samples/LVH) — X-rays with confirmed LVH  
- 📂 [samples/No_LVH](samples/No_LVH) — X-rays without LVH

👉 Download any image and upload it in the [Live Demo](https://huggingface.co/spaces/Beyonder016/lvh-gradio) interface to see predictions and Grad-CAM heatmaps.

---

## ⚙️ How It Works

1. Upload a `.jpg` or `.png` chest X-ray  
2. The model predicts: **LVH** or **No LVH**  
3. A heatmap highlights the region used for the prediction

---

## 💻 Run Locally

```bash
git clone https://github.com/Beyonder016/lvh-gradio.git
cd lvh-gradio
pip install -r requirements.txt
python app.py
```
## 🛠 Built With
-PyTorch & torchvision
-Grad-CAM (model explainability)
-Gradio (interactive web UI)
-Hugging Face Spaces (deployment)

🙌 Author
Made with ❤️ by @Beyonder016
