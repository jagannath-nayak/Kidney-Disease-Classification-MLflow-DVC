import streamlit as st
from fpdf import FPDF
import base64

st.set_page_config(page_title="Model & Data Cards", page_icon="📄")

model_card_content = """
### 🧠 Model Card

**Model Name:** Kidney Tumor Classifier  
**Model Type:** CNN (Convolutional Neural Network)  
**Framework:** TensorFlow / Keras  
**Architecture:** Custom CNN with Conv2D, MaxPool, Dense layers  
**Trained On:** Kidney CT Scans (Tumor vs. Normal)  
**Accuracy:** 95%  
**Evaluation Metrics:** Accuracy, Precision, Recall  

#### ✅ Intended Use
- Assist radiologists in identifying tumors in kidney CT scans.
- Educational purposes and research.

#### ⚠️ Limitations
- Not intended for sole diagnostic decisions.
- May not generalize to unseen CT protocols or devices.
- Biased if used outside the training demographic.
"""

data_card_content = """
### 📊 Data Card

**Dataset:** Kidney Tumor Detection Dataset  
**Source:** Private Collection  
**Total Samples:** ~2000 images  
**Classes:** Tumor (1), Normal (0)  
**Input Shape:** 224 x 224 x 3  
**Image Modality:** CT/MRI  

#### ⚠️ Data Bias Notes
- Majority of scans are from adults.
- May lack diversity across ethnicity.
- Needs testing on pediatric populations.
"""

# Display content
st.markdown(model_card_content)
st.markdown("---")
st.markdown(data_card_content)

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Kidney Tumor Classifier - Model & Data Card", ln=True, align="C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_section(self, title, body):
        self.chapter_title(title)
        self.chapter_body(body)

model_card_plain = """Model Name: Kidney Tumor Classifier
Model Type: CNN (Convolutional Neural Network)
Framework: TensorFlow / Keras
Architecture: Custom CNN with Conv2D, MaxPool, Dense layers
Trained On: Kidney CT Scans (Tumor vs. Normal)
Accuracy: 95%
Evaluation Metrics: Accuracy, Precision, Recall

Intended Use:
- Assist radiologists in identifying tumors in kidney CT scans.
- Educational purposes and research.

Limitations:
- Not intended for sole diagnostic decisions.
- May not generalize to unseen CT protocols or devices.
- Biased if used outside the training demographic."""

data_card_plain = """Dataset: Kidney Tumor Detection Dataset
Source: Private Collection
Total Samples: ~2000 images
Classes: Tumor (1), Normal (0)
Input Shape: 224 x 224 x 3
Image Modality: CT/MRI

Data Bias Notes:
- Majority of scans are from adults.
- May lack diversity across ethnicity.
- Needs testing on pediatric populations."""

# Generate PDF
pdf = PDF()
pdf.add_page()
pdf.add_section("Model Card", model_card_plain)
pdf.add_section("Data Card", data_card_plain)

# Save locally
pdf_path = "model_data_card.pdf"
pdf.output(pdf_path)

# Show download link
with open(pdf_path, "rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    st.markdown(
        f'<a href="data:application/pdf;base64,{base64_pdf}" download="Model_Data_Card.pdf">📥 Download Model & Data Card as PDF</a>',
        unsafe_allow_html=True
    )
