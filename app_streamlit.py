#  FarmScout: AI-based Rice Health Detection
# Author: Chynna Rangas
# Compatible with TensorFlow 2.15 / Keras 3.4
# Streamlit Web App

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------- CONFIGURATION ----------
MODEL_PATH = "model/RiceHealthAnalysis_model.h5"
LABELS_PATH = "model/RiceHealthAnalysis_labels.txt"

st.set_page_config(
    page_title=" FarmScout: Rice Health Detection",
    page_icon="🌱",
    layout="centered"
)

st.title(" FarmScout: Rice Health Analysis")
st.markdown("### Upload a rice leaf image to analyze its health condition using your AI model.")

# ---------- SAFE MODEL LOADING ----------
@st.cache_resource
def load_rice_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r") as f:
                labels = [line.strip() for line in f.readlines()]
        else:
            labels = ["Healthy", "Unhealthy"]
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, []

model, labels = load_rice_model()

if model is None:
    st.stop()

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader("Upload a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------- PREPROCESS IMAGE ----------
    img_size = model.input_shape[1:3]  # dynamically detect input size
    img_resized = image.resize(img_size)
    img_array = np.array(img_resized).astype("float32") / 255.0  # normalization (0–1)
    img_array = np.expand_dims(img_array, axis=0)

    # ---------- PREDICTION ----------
    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing image... please wait ⏳"):
            preds = model.predict(img_array)
            pred_idx = int(np.argmax(preds))
            confidence = float(np.max(preds) * 100)
            label = labels[pred_idx] if pred_idx < len(labels) else "Unknown"

        # ---------- DISPLAY RESULT ----------
        st.subheader("AI Analysis Result:")
        if "healthy" in label.lower():
            st.success(f" The rice leaf is **Healthy** with {confidence:.2f}% confidence.")
        elif "unhealthy" in label.lower() or "diseased" in label.lower():
            st.error(f"The rice leaf is **Unhealthy** with {confidence:.2f}% confidence.")
        else:
            st.warning(f"Classified as **{label}** ({confidence:.2f}% confidence).")

        st.progress(int(confidence))

# ---------- SIDEBAR INFO ----------
st.sidebar.header("📘About FarmScout")
st.sidebar.markdown("""
**FarmScout** is an AI-based project designed to analyze rice plant health using
a Convolutional Neural Network (CNN) trained via Google Colab.

**Developed by:**  
👩‍💻 *Chynna Rangas*  
🏫 *National University - Philippines*

**Model:** RiceHealthAnalysis_model.h5  
**Frameworks:** TensorFlow · Streamlit · NumPy · Pillow
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Capstone 2025 — FarmScout: Real-Time Farm Insights from an IoT-Enabled Aerial Platform*")