import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Patch Keras Dense layer deserialization issue (quantization_config mismatch)
import keras
_original_dense_init = keras.layers.Dense.__init__
def _patched_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    return _original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = _patched_dense_init

# ── Config ──────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE = (224, 224)
MODEL_PATH = "tl_model.keras"   # place this file next to app.py

CLASS_INFO = {
    "Glioma": {
        "color": "#e74c3c",
        "desc": "A tumor that originates in the glial cells of the brain or spine. Can be low- or high-grade.",
    },
    "Meningioma": {
        "color": "#e67e22",
        "desc": "A typically slow-growing tumor that forms in the meninges (brain/spinal cord membranes). Usually benign.",
    },
    "No Tumor": {
        "color": "#27ae60",
        "desc": "No tumor detected in the MRI scan.",
    },
    "Pituitary": {
        "color": "#8e44ad",
        "desc": "A tumor that forms in the pituitary gland. Most are benign adenomas.",
    },
}

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; text-align: center; margin-bottom: 0; }
    .subtitle   { text-align: center; color: #888; margin-top: 4px; margin-bottom: 2rem; }
    .result-box { padding: 1.2rem 1.5rem; border-radius: 10px; margin-top: 1rem; }
    .conf-label { font-size: 0.85rem; color: #aaa; margin-bottom: 2px; }
    .disclaimer { font-size: 0.75rem; color: #888; text-align: center; margin-top: 2rem;
                  border-top: 1px solid #333; padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧠 Brain Tumor Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a brain MRI scan — EfficientNetB0 transfer learning model</p>', unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file `{MODEL_PATH}` not found. Place it in the same directory as app.py.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"],
    help="Accepts .jpg, .jpeg, .png",
)

if uploaded:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(uploaded, caption="Uploaded MRI", use_container_width=True)

    # Preprocess
    img = Image.open(uploaded).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)          # 0–255, model handles rescaling
    img_batch = np.expand_dims(img_array, axis=0)        # (1, 224, 224, 3)

    # Inference
    with st.spinner("Running inference…"):
        preds = model.predict(img_batch, verbose=0)[0]   # shape (4,)

    pred_idx   = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf  = float(preds[pred_idx]) * 100
    info       = CLASS_INFO[pred_class]

    with col2:
        st.markdown(f"""
        <div class="result-box" style="background-color:{info['color']}22; border-left: 4px solid {info['color']};">
            <div style="font-size:1.05rem; font-weight:600; color:{info['color']};">Prediction</div>
            <div style="font-size:2rem; font-weight:700; margin: 6px 0;">{pred_class}</div>
            <div style="font-size:1.1rem; font-weight:600;">Confidence: {pred_conf:.1f}%</div>
            <div style="margin-top: 10px; font-size: 0.9rem; color:#ccc;">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Class Probabilities")
        sorted_idx = np.argsort(preds)[::-1]
        for i in sorted_idx:
            bar_color = CLASS_INFO[CLASS_NAMES[i]]["color"]
            st.markdown(f'<div class="conf-label">{CLASS_NAMES[i]}</div>', unsafe_allow_html=True)
            st.progress(float(preds[i]), text=f"{preds[i]*100:.1f}%")

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
**Model:** EfficientNetB0 (transfer learning)  
**Input:** 224 × 224 RGB MRI scans  
**Classes:** 4  
- Glioma  
- Meningioma  
- No Tumor  
- Pituitary  

**Dataset:** Brain Tumor MRI (Kaggle)  
~5,600 training images  
""")
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("`TensorFlow · Keras · Streamlit`")

st.markdown(
    '<p class="disclaimer">⚠️ For educational purposes only. Not a substitute for medical diagnosis.</p>',
    unsafe_allow_html=True,
)
