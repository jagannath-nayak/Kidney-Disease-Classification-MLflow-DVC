import streamlit as st
from features.functions import load_lottie_file
import streamlit_lottie as st_lottie

# App config
st.set_page_config(
    page_title="Kidney Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize state
if 'all_predictions' not in st.session_state:
    st.session_state['all_predictions'] = []

# Home function
def home():
    st.header("🧠 Kidney Tumor Classifier", divider='rainbow')

    # Section 1: About
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.subheader("About the App", divider='rainbow')
            st.markdown("""
This app is a deep learning–based **Kidney Tumor Classification Tool** that classifies CT/MRI images as **Tumor** or **Normal**.

It helps radiologists, researchers, and healthcare providers get fast predictions, understand the model’s reasoning, and ensure fair treatment across patient demographics.
            """)
        with right:
            banner = load_lottie_file("animations/banner..json")
            if banner:
                st_lottie.st_lottie(banner, loop=True, width=500, height=400)

    # Section 2: Features
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.subheader("Key Features", divider='rainbow')
            st.markdown("""
- 📤 Upload CT or MRI image for analysis  
- 🔍 Predict: Tumor or Normal  
- 🧠 Model Explainability with Grad-CAM fallback  
- 📊 Bias Dashboard for fairness audit  
- 🧾 Downloadable PDF prediction report  
- 📄 Transparent Model & Data Card
            """)
        with right:
            features_anim = load_lottie_file("animations/analyze.json")
            if features_anim:
                st_lottie.st_lottie(features_anim, loop=True, width=500, height=400)

    # Section 3: Benefits
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.subheader("Benefits", divider='rainbow')
            st.markdown("""
- ✅ Quick and accurate diagnosis support  
- ✅ Increases trust with visual explanations  
- ✅ Promotes ethical AI with bias checks  
- ✅ Easily shareable reports with physicians  
- ✅ Trained with 95% accuracy for reliability
            """)
        with right:
            benefit_anim = load_lottie_file("animations/success..json")
            if benefit_anim:
                st_lottie.st_lottie(benefit_anim, loop=True, width=500, height=400)

    # Section 4: How It Solves the Problem
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.subheader("How It Solves the Problem", divider='rainbow')
            st.markdown("""
- 🧠 Uses VGG16 deep learning architecture  
- 🔍 Grad-CAM for model interpretability  
- 📊 Bias analysis by gender, age group, etc.  
- 📄 Generates professional, downloadable reports  
- ⚕️ Supports early detection & informed treatment
            """)
        with right:
            solve_anim = load_lottie_file("animations/tumor_scan.json")
            if solve_anim:
                st_lottie.st_lottie(solve_anim, loop=True, width=500, height=400)

    # Section 5: FAQs
    with st.container(border=True):
        st.subheader("FAQs", divider='rainbow')
        with st.expander("📌 What is this app for?"):
            st.write("It predicts whether a kidney image shows signs of a tumor or not using a deep learning model.")
        with st.expander("📌 How accurate is it?"):
            st.write("The model is trained with 95% accuracy on a labeled dataset.")
        with st.expander("📌 Can I trust the predictions?"):
            st.write("Yes. Grad-CAM explainability helps you see where the model focused during prediction.")
        with st.expander("📌 Do I need medical expertise?"):
            st.write("Not necessarily. The app simplifies interpretation, but final decisions should be taken by professionals.")

# Navigation setup
pg = st.navigation([
    st.Page(title="Home", page=home, icon="🏠"),
    st.Page(title="Kidney Tumor Predictor", page="features/0-Kidney-Tumor-Predictor.py", icon="🧪"),
    st.Page(title="Bias Dashboard", page="features/1-Bias-Dashboard.py", icon="⚖️"),
    st.Page(title="Model & Data Card", page="features/2-Model-Data-Card.py", icon="📄"),
])
pg.run()
