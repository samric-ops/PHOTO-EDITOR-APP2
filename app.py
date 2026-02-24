import io
import os
import streamlit as st
from PIL import Image
import numpy as np

# --- Helper Functions ---
def get_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        st.error(f"OpenCV failed to load. Technical detail: {repr(e)}")
        st.stop()

@st.cache_resource(show_spinner="Loading AI Models... (This takes a moment)")
def load_models():
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer

    # Initialize RealESRGAN (Background Upsampler)
    # tile=100 is used to save RAM on Streamlit Cloud
    sr_upsampler = RealESRGANer(
        scale=4,
        model_path=None,
        model="RealESRGAN_x4plus",
        tile=100, 
        tile_pad=10, 
        pre_pad=0,
        half=False, # CPUs don't support half-precision (FP16)
        device='cpu'
    )

    # Initialize GFPGAN (Face Restorer)
    face_enhancer = GFPGANer(
        model_path=None,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=sr_upsampler,
        device='cpu'
    )
    return face_enhancer

def process_image(img_np, upscale_factor, denoise_strength, face_enhancer):
    cv2 = get_cv2()
    
    # Optional Denoising
    if denoise_strength > 0:
        h = int(denoise_strength * 10)
        img_np = cv2.fastNlMeansDenoisingColored(img_np, None, h, h, 7, 21)

    # Enhance Face and Background
    _, _, restored_img = face_enhancer.enhance(
        img_np,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    
    return restored_img

# --- UI Layout ---
st.set_page_config(page_title="AI Photo Enhancer", page_icon="✨")
st.title("✨ AI Photo Enhancer")
st.markdown("Restore faces and upscale images using GFPGAN & Real-ESRGAN.")

with st.sidebar:
    st.header("Settings")
    upscale = st.selectbox("Upscale Factor", [2, 4], index=0)
    denoise = st.slider("Denoise Strength", 0.0, 1.0, 0.3)
    st.info("Note: Processing takes ~30-60 seconds on CPU.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load Image
    input_image = Image.open(uploaded_file).convert("RGB")
    input_np = np.array(input_image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="Original Image", use_column_width=True)

    if st.button("Magic Enhance ✨"):
        try:
            face_enhancer = load_models()
            with st.spinner("Processing... please wait."):
                result_np = process_image(input_np, upscale, denoise, face_enhancer)
                
                result_pil = Image.fromarray(result_np)
                with col2:
                    st.image(result_pil, caption="Enhanced Image", use_column_width=True)
                
                # Download Button
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button(
                    label="Download Enhanced Image",
                    data=buf.getvalue(),
                    file_name="enhanced_photo.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("If the app crashed, the image might be too large for the free tier RAM.")
