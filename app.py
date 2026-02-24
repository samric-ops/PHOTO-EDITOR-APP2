import os
# --- EMERGENCY FIX FOR BASICSR ---
os.environ['BASICSR_JIT'] = 'True'
os.environ['BASICSR_EXT'] = 'True'

import io
import streamlit as st
from PIL import Image
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- System Check ---
try:
    import cv2
    st.success("✓ OpenCV loaded successfully")
except ImportError as e:
    st.error(f"OpenCV System Error: {e}. Please ensure 'packages.txt' exists and Reboot.")
    st.stop()

# --- Functions ---
def resize_for_ram(img_pil, max_size=800):  # Reduced from 1000 to 800 for better RAM management
    """Resize image if too large to prevent 1GB RAM crash."""
    w, h = img_pil.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return img_pil.resize(new_size, Image.LANCZOS)
    return img_pil

@st.cache_resource(show_spinner="Loading AI Models... (this may take 2-3 minutes)")
def load_models():
    try:
        from gfpgan import GFPGANer
        from realesrgan import RealESRGANer
        import torch
        
        st.info("Loading Real-ESRGAN upsampler...")
        # Background Upsampler (Real-ESRGAN)
        sr_upsampler = RealESRGANer(
            scale=2,
            model_path=None,
            model="RealESRGAN_x4plus",
            tile=50,  # Reduced from 100 for better memory management
            tile_pad=10,
            pre_pad=0,
            half=False,
            device='cpu'
        )
        
        st.info("Loading GFPGAN face enhancer...")
        # Face Restorer (GFPGAN)
        face_enhancer = GFPGANer(
            model_path=None,
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=sr_upsampler,
            device='cpu'
        )
        
        st.success("✓ Models loaded successfully!")
        return face_enhancer
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None

# --- UI ---
st.set_page_config(page_title="AI Photo Enhancer", page_icon="✨")
st.title("✨ AI Photo Enhancer")
st.markdown("Upload a photo to enhance it using AI (GFPGAN + Real-ESRGAN)")

with st.sidebar:
    st.header("Settings")
    st.info("ℹ️ Free tier is limited to 1GB RAM. Processing large images may take time.")
    st.warning("⚠️ For best results, use images with faces")
    
    with st.expander("Advanced Options"):
        denoise = st.slider("Denoise Strength", 0.0, 1.0, 0.3, disabled=True)
        st.caption("Denoise feature coming soon")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    # 1. Load and Resize if necessary
    try:
        img_pil = Image.open(uploaded).convert("RGB")
        original_size = img_pil.size
        img_pil = resize_for_ram(img_pil)
        
        if img_pil.size != original_size:
            st.info(f"Image resized from {original_size} to {img_pil.size} for processing")
        
        img_np = np.array(img_pil)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_pil, caption="Original (Resized for processing)", use_column_width=True)

        if st.button("✨ Start Enhancement", type="primary"):
            try:
                with st.spinner("Loading models..."):
                    enhancer = load_models()
                
                if enhancer is None:
                    st.stop()
                
                with st.spinner("Processing image... Do not refresh the page."):
                    # Process
                    _, _, out_np = enhancer.enhance(
                        img_np,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                    
                    out_pil = Image.fromarray(out_np)
                    
                    with col2:
                        st.image(out_pil, caption="Enhanced Image", use_column_width=True)

                    # Download button
                    buf = io.BytesIO()
                    out_pil.save(buf, format="PNG")
                    st.download_button(
                        "⬇️ Download Result",
                        buf.getvalue(),
                        "enhanced.png",
                        "image/png",
                        use_container_width=True
                    )

            except MemoryError:
                st.error("Out of Memory! Try uploading a smaller image.")
            except Exception as e:
                st.error(f"Error during enhancement: {str(e)}")
                st.info("Tip: Try a smaller image or restart the app")
                
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
