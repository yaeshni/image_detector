import streamlit as st
import joblib
import cv2
import numpy as np
import tempfile
import os
import time
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    clf = joblib.load(r"C:\Users\Asus\OneDrive\Desktop\data\model.pkl")
    return cnn, clf

cnn, clf = load_models()

# ── Prediction function ───────────────────────────────────────────────────────
def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    img = preprocess_input(np.expand_dims(img, axis=0))
    start = time.time()
    emb = cnn.predict(img, verbose=0).flatten().reshape(1, -1)
    score = float(clf.predict_proba(emb)[0][1])
    latency_ms = (time.time() - start) * 1000
    return score, latency_ms

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spot the Fake Photo", page_icon="🔍", layout="centered")

st.title("Spot the Fake Photo")
st.markdown("Upload an image to check if it is a **real photo** or a **photo of a screen**.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Save to temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analysing..."):
        score, latency = predict(tmp_path)

    os.unlink(tmp_path)  # clean up temp file

    # ── Result display ────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Score", value=f"{score:.4f}", help="0 = real photo, 1 = screen recapture")

    with col2:
        st.metric(label="Latency", value=f"{latency:.1f} ms")

    if score > 0.5:
        st.error(f"SCREEN RECAPTURE (score: {score:.4f})")
        st.markdown("This looks like a **photo of a screen or printout**, not a real photo.")
    else:
        st.success(f"REAL PHOTO (score: {score:.4f})")
        st.markdown("This looks like a **genuine real-world photo**.")

    st.progress(score, text=f"Confidence toward SCREEN: {score:.2%}")

st.markdown("---")
st.caption("Model: MobileNetV2 (frozen) + Logistic Regression  |  Accuracy: 95%  |  Built by Aeshni")