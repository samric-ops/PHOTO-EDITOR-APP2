import os
# --- EMERGENCY FIX FOR BASICSR ---
os.environ['BASICSR_JIT'] = 'True'
os.environ['BASICSR_EXT'] = 'True'

import io
import streamlit as st
from PIL import Image
import numpy as np

# --- System Check ---
try:
    import cv2
except ImportError:
    st.error("OpenCV System Error: Please ensure 'packages.txt' exists and Reboot.")
    st.stop()

# --- Functions ---
def resize_for_ram(img_pil, max_size=1000):
    """Liliitan ang image kung masyadong malaki para hindi mag-crash ang 1GB RAM."""
    w, h = img_pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return img_pil.resize(new_size, Image.LANCZOS)
    return img_pil

@st.cache_resource(show_spinner="Loading AI Models... (1-2 minutes)")
def load_models():
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer

    # Background Upsampler (Real-ESRGAN)
    # tile=100 helps process large images in small chunks to save RAM
    sr_upsampler = RealESRGANer(
        scale=2, 
        model_path=None,
        model="RealESRGAN_x4plus",
        tile=100, 
        tile_pad=10, 
        pre_pad=0,
        half=False, # CPU doesn't support FP16
        device='cpu'
    )

    # Face Restorer (GFPGAN)
    face_enhancer = GFPGANer(
        model_path=None,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=sr_upsampler,
        device='cpu'
    )
    return face_enhancer

# --- UI ---
st.set_page_config(page_title="AI Photo Enhancer", page_icon="✨")
st.title("✨ AI Photo Enhancer")

with st.sidebar:
    st.header("Settings")
    denoise = st.slider("Denoise Strength", 0.0, 1.0, 0.3)
    st.warning("Note: Free tier is limited to 1GB RAM. Processing large images may take time.")

uploaded = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded:
    # 1. Load and Resize if necessary
    img_pil = Image.open(uploaded).convert("RGB")
    img_pil = resize_for_ram(img_pil)
    img_np = np.array(img_pil)

    st.image(img_pil, caption="Original (Resized for processing)", use_column_width=True)

    if st.button("Start Enhancement"):
        try:
            enhancer = load_models()
            with st.spinner("Processing... Do not refresh."):
                # Process
                _, _, out_np = enhancer.enhance(
                    img_np, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                
                out_pil = Image.fromarray(out_np)
                st.image(out_pil, caption="Enhanced Image", use_column_width=True)

                # Download
                buf = io.BytesIO()
                out_pil.save(buf, format="PNG")
                st.download_button("⬇️ Download Result", buf.getvalue(), "enhanced.png", "image/png")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Tip: If it says 'Out of Memory', try a smaller image.")
