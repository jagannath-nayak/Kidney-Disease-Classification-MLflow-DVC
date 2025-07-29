import streamlit as st
from features.functions import load_lottie_file
import streamlit_lottie as st_lottie

st.set_page_config(
    page_title="Kidney Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def safe_lottie(path: str):
    try:
        return load_lottie_file(path)
    except Exception:
        return None

def home():
    st.header("🧠 Kidney Tumor Classifier", divider='rainbow')

    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.subheader("About the App", divider='rainbow')
            st.markdown(
                """
This Streamlit app is a deep learning-powered **Kidney Tumor Classification Tool**.

**What you can do here:**
- 📤 Upload a CT/MRI image
- 🔍 Get a prediction: **Tumor** or **Normal**
- 🧠 See **explanations** (Integrated Gradients / Grad‑CAM fallback)
- 🧾 Download a **Prediction PDF report**
- ⚖️ Explore a **Bias Dashboard** across demographics
- 📄 View & download the **Model & Data Card**
- 🎯 Trained model with **95% accuracy**
                """
            )
        with right:
            banner = safe_lottie("animations/banner..json")
            if banner:
                st_lottie.st_lottie(banner, loop=True, width=500, height=350)
            else:
                st.info("Add a Lottie file at `animations/banner.json` to show an animation here.")

    with st.container(border=True):
        st.subheader("🔭 Visuals used in the app", divider='rainbow')
        c1, c2 = st.columns(2)
        with c1:
            analyze = safe_lottie("animations/analyze.json")
            if analyze:
                st_lottie.st_lottie(analyze, loop=True, height=250)
            else:
                st.caption("Missing: animations/analyze.json")

            success = safe_lottie("animations/success..json")
            if success:
                st_lottie.st_lottie(success, loop=True, height=250)
            else:
                st.caption("Missing: animations/success.json")

        with c2:
            tumor_scan = safe_lottie("animations/tumor_scan.json")
            if tumor_scan:
                st_lottie.st_lottie(tumor_scan, loop=True, height=250)
            else:
                st.caption("Missing: animations/tumor_scan.json")


            if banner:
                st_lottie.st_lottie(banner, loop=True, height=250)

    st.success("✅ Navigate using the sidebar (left) to try the Predictor, Bias Dashboard, and Model/Data Card.")

pg = st.navigation([
    st.Page(title="Home", page=home, icon="🏠"),
    st.Page(title="Kidney Tumor Predictor", page="features/0-Kidney-Tumor-Predictor.py", icon="🧪"),
    st.Page(title="Bias Dashboard", page="features/1-Bias-Dashboard.py", icon="⚖️"),
    st.Page(title="Model & Data Card", page="features/2-Model-Data-Card.py", icon="📄"),
])

pg.run()
