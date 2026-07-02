# Brain Tumor Classifier — Streamlit App

EfficientNetB0 transfer learning model classifying brain MRI scans into 4 categories:
**Glioma · Meningioma · No Tumor · Pituitary**

---

## Setup

### 1. Add your model file
Place `tl_model.keras` (saved from Colab via Google Drive) in this folder, next to `app.py`.

> **Why not `cnn_model.keras`?**  
> The TL model (EfficientNetB0) significantly outperforms the custom CNN. Only one model is loaded to keep cold-start time low on Streamlit Cloud.

### 2. Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 3. Deploy to Streamlit Cloud
1. Push this folder (with `tl_model.keras` included) to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → point to `app.py`.
3. Done.

> ⚠️ `tl_model.keras` is ~17 MB — fine for GitHub. If you ever switch to a larger model, use Git LFS.

---

## File structure
```
brain_tumor_app/
├── app.py
├── requirements.txt
├── README.md
└── tl_model.keras   ← you add this
```

---

## Notes
- Model input: raw 0–255 pixel values. EfficientNetB0 handles its own normalization internally.
- Data augmentation layers are inside the model but have no effect at inference time (Keras disables them automatically).
- `tensorflow-cpu` is used in requirements to avoid the ~1 GB GPU build on Streamlit Cloud (no GPU available there anyway).
