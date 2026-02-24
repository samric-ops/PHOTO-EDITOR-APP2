import os
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Test App", page_icon="✅")
st.title("✅ Test App")
st.write("If you can see this, basic imports work!")

# Test OpenCV
try:
    import cv2
    st.success("✓ OpenCV loaded successfully")
except Exception as e:
    st.error(f"OpenCV error: {e}")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image")
