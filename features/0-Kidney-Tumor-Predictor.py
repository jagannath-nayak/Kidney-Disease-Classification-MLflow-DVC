import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import uuid

from src.cnnClassifier.utils.gradcam_utils import explain_image, overlay_heatmap_on_image
from fpdf import FPDF

st.header("🧪 Kidney Tumor Prediction", divider="rainbow")

MODEL_PATH = "model/model.h5"
MODEL = load_model(MODEL_PATH, compile=False)

UPLOAD_DIR = "uploaded"
PREDICTION_HISTORY = []

os.makedirs(UPLOAD_DIR, exist_ok=True)

def generate_pdf_report(label, image_path, save_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Kidney Tumor Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.ln(5)

    pdf.image(image_path, x=10, y=40, w=100)
    pdf.output(save_path)

uploaded_file = st.file_uploader("Upload a kidney CT/MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(file_path, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        # Preprocess
        test_image = image.load_img(file_path, target_size=(224, 224))
        test_array = image.img_to_array(test_image) / 255.0
        img_array = np.expand_dims(test_array, axis=0).astype(np.float32)
        img_tensor = tf.convert_to_tensor(img_array)

        # Predict
        preds = MODEL.predict(img_tensor)
        if preds.shape[-1] == 1:
            predicted_index = int(preds[0][0] > 0.5)
        else:
            predicted_index = int(np.argmax(preds[0]))

        class_map = {0: "Tumor", 1: "Normal"}
        label = class_map.get(predicted_index, "Unknown")

        st.success(f"✅ Prediction: **{label}**")

        # Explainability
        try:
            heatmap, pred_index, method_used = explain_image(img_tensor, MODEL)
            overlay = overlay_heatmap_on_image(heatmap, file_path, alpha=0.45)
            st.image(overlay, caption=f"Explanation using {method_used.upper()}", use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not generate explanation: {str(e)}")

        # PDF report
        pdf_path = f"{file_path}.pdf"
        generate_pdf_report(label=label, image_path=file_path, save_path=pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Report (PDF)", f, file_name="prediction_report.pdf", mime="application/pdf")
