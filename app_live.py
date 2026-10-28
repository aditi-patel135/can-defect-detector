import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "defect_detector_cans.keras"
# Training labels: 0 = Normal, 1 = Defective. Model output = P(Defective).

st.set_page_config(page_title="Can Defect Detector", page_icon="🥫", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict(model, img):
    _, h, w, _ = model.input_shape
    img = img.convert("RGB").resize((w, h))
    arr = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)  # no /255
    return float(model.predict(arr, verbose=0)[0][0])


def show_result(img, name, p_defective, threshold):
    is_normal = p_defective < threshold
    label = "Normal" if is_normal else "Defective"
    confidence = 1 - p_defective if is_normal else p_defective
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption=name, width="stretch")
    with col2:
        if is_normal:
            st.success(f"✅ {label}")
        else:
            st.error(f"❌ {label}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(p_defective, text=f"P(Defective) = {p_defective:.3f}")


st.title("🥫 Aluminum Can Defect Detector")

with st.sidebar:
    threshold = st.slider("Defective threshold", 0.05, 0.95, 0.50, 0.05,
                          help="If P(Defective) is at or above this, the can is labeled Defective.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load {MODEL_PATH}. Put it in the same folder as this file.\n\n{e}")
    st.stop()

tab_cam, tab_up = st.tabs(["📷 Live camera", "📁 Upload photos"])

with tab_cam:
    st.caption("Click Allow when the browser asks for camera permission, then take a photo.")
    shot = st.camera_input("Take a photo of the can")
    if shot is not None:
        img = Image.open(shot)
        show_result(img, "Camera photo", predict(model, img), threshold)

with tab_up:
    files = st.file_uploader("Choose can image(s)", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True)
    for f in files or []:
        img = Image.open(f)
        show_result(img, f.name, predict(model, img), threshold)
        st.divider()