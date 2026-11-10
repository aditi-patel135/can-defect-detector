import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Defect Detector", layout="wide")
st.title("Defective Product Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("defect_detector.keras")

model = load_model()

# sidebar controls
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Defect threshold", 0.1, 0.9, 0.5, 0.05)
st.sidebar.caption("Lower = catches more defects, but more false alarms")

# counters that survive reruns
for k in ("total", "normal", "defective"):
    st.session_state.setdefault(k, 0)

def predict(pil_img):
    img = pil_img.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype="float32"), 0)
    return float(model.predict(arr, verbose=0)[0][0])   # probability of Defective

left, right = st.columns(2)
with left:
    st.subheader("Capture or upload")
    cam = st.camera_input("Take a photo of the product")
    up = st.file_uploader("...or upload an image", type=["jpg", "jpeg", "png", "bmp"])
    source = cam or up

with right:
    st.subheader("Result")
    if source is not None:
        img = Image.open(source)
        p = predict(img)
        st.image(img, use_container_width=True)
        if p >= threshold:
            st.error(f"DEFECTIVE  (confidence {p*100:.1f}%)")
            label = "defective"
        else:
            st.success(f"NORMAL  (confidence {(1-p)*100:.1f}%)")
            label = "normal"
        if st.button("Add this to the inspection log"):
            st.session_state.total += 1
            st.session_state[label] += 1

c1, c2, c3 = st.columns(3)
c1.metric("Total inspected", st.session_state.total)
c2.metric("Normal", st.session_state.normal)
c3.metric("Defective", st.session_state.defective)