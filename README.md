# 🧠 Brain Tumor MRI Classifier

Classifies brain MRI scans into **4 categories** using EfficientNetB0 transfer learning, deployed as an interactive Streamlit web app.

**Classes:** Glioma · Meningioma · No Tumor · Pituitary

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://brain-tumor-mri-classifier-kz3gw8tjondgl5apperxjye.streamlit.app/)

---

## Demo

![App Screenshot](demo.png)

---

## Model

| Detail | Value |
|---|---|
| Architecture | EfficientNetB0 (ImageNet weights, fine-tuned) |
| Input size | 224 × 224 RGB |
| Output | 4-class softmax |
| Dataset | [Brain Tumor MRI — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (~5,600 train images) |
| Preprocessing | Raw 0–255 pixel values (EfficientNetB0 handles internal normalization) |

> A custom CNN baseline was trained alongside EfficientNetB0. The transfer learning model significantly outperformed it and is the only model served in the app to keep cold-start time low on Streamlit Cloud.

---

## Project Structure

```
brain_tumor_app/
├── app.py               # Streamlit app
├── requirements.txt
├── README.md
└── tl_model.keras       # EfficientNetB0 model (add manually — see setup)
```

---

## Run Locally

```bash
git clone https://github.com/<your-username>/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier

pip install -r requirements.txt

# Place tl_model.keras in the root directory (download from Google Drive / Colab export)
streamlit run app.py
```

> **Note:** `tl_model.keras` is ~17 MB and tracked in Git directly. If you switch to a larger backbone, use [Git LFS](https://git-lfs.com/).

---

## Deploy on Streamlit Cloud

1. Push the repo (with `tl_model.keras` included) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → point to `app.py`.
3. Done — no extra configuration needed.

---

## Tech Stack

`TensorFlow` · `Keras` · `Streamlit` · `Pillow` · `NumPy`

---

> ⚠️ **Disclaimer:** For educational purposes only. Not a substitute for clinical medical diagnosis.
