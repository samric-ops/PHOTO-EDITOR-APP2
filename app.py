import io
import os
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image
import numpy as np

# --- lazy import for cv2 ---
def get_cv2():
    try:
        import cv2
        return cv2
    except Exception as e:
        st.error(
            "OpenCV (`cv2`) failed to import in this environment.\n\n"
            "Make sure `runtime.txt` is `3.11` and `requirements.txt` pins "
            "`opencv-python-headless==4.5.5.64`. After pushing, go to **Manage app → Reboot** "
            "and if needed **Settings → Advanced → Clear cache**.\n\n"
            f"Technical detail: {repr(e)}"
        )
        st.stop()

def pil_to_np(img: Image.Image) -> np.ndarray:
    # RGB NumPy array
    return np.array(img.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)

def apply_tone_pil(img: Image.Image, brightness=0, contrast=0, saturation=0) -> Image.Image:
    from PIL import ImageEnhance
    # Convert 0-centered sliders to enhancer multipliers
    out = img
    if brightness != 0:
        out = ImageEnhance.Brightness(out).enhance(1 + brightness / 100.0)
    if contrast != 0:
        out = ImageEnhance.Contrast(out).enhance(1 + contrast / 100.0)
    if saturation != 0:
        out = ImageEnhance.Color(out).enhance(1 + saturation / 100.0)
    return out

def smooth_background_np(img_np: np.ndarray, strength=10) -> np.ndarray:
    # optional smoothing using OpenCV; called lazily so import only happens if needed
    cv2 = get_cv2()
    s = max(5, int(strength))
    return cv2.bilateralFilter(img_np, d=9, sigmaColor=s*3, sigmaSpace=s)

@st.cache_resource(show_spinner="Loading enhancement models…")
def load_models():
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer

    cache_dir = Path.home() / ".cache" / "ai-photo-enhancer"
    os.makedirs(cache_dir, exist_ok=True)

    sr_upsampler = RealESRGANer(
        scale=4,
        model_path=None,
        model="RealESRGAN_x4plus",
        tile=200, tile_pad=10, pre_pad=0,
        half=True
    )

    face_enhancer = GFPGANer(
        model_path=None,
        upscale=4,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=sr_upsampler
    )
    return face_enhancer, sr_upsampler

def enhance_image(
    img_np: np.ndarray,
    upscale: int = 2,
    denoise_strength: float = 0.5,
    bg_smooth_strength: int = 10,
    tone: Optional[dict] = None
) -> np.ndarray:
    cv2 = get_cv2()
    face_enhancer, _ = load_models()

    # Pre-denoise (OpenCV)
    if denoise_strength > 0:
        img_np = cv2.fastNlMeansDenoisingColored(
            img_np, None,
            5 + int(denoise_strength * 5),
            5 + int(denoise_strength * 5),
            7, 21
        )

    # GFPGAN returns restored + upscaled (≈4×)
    _, _, restored = face_enhancer.enhance(
        img_np,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    out = restored

    # Downscale to 2× if selected
    if upscale == 2:
        h, w = out.shape[:2]
        out = cv2.resize(out, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)

    # Light background smoothing
    if bg_smooth_strength > 0:
        out = smooth_background_np(out, strength=bg_smooth_strength)

    # Tone in Pillow (to avoid highgui)
    out_pil = apply_tone_pil(
        Image.fromarray(out),
        brightness=tone.get("brightness", 0) if tone else 0,
        contrast=tone.get("contrast", 8) if tone else 0,
        saturation=tone.get("saturation", 6) if tone else 0
    )
    return np.array(out_pil)

# -------------------- UI --------------------

st.set_page_config(page_title="AI Photo Enhancer", page_icon="✨", layout="centered")
st.title("✨ AI Photo Enhancer")
st.caption("Upscale • Denoise • Restore • Natural tones")

with st.sidebar:
    st.header("Quality Controls")
    upscale = st.selectbox("Upscale factor", [2, 4], index=0)
    denoise = st.slider("Denoise strength", 0.0, 1.0, 0.5, 0.05)
    bg_smooth = st.slider("Background smoothing", 0, 30, 10)
    st.subheader("Tone")
    brightness = st.slider("Brightness", -50, 50, 0)
    contrast   = st.slider("Contrast",   -50, 50, 8)
    saturation = st.slider("Saturation", -30, 30, 6)

uploaded = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if not uploaded:
    st.info("Upload an image to start.")
    st.stop()

orig_pil = Image.open(uploaded).convert("RGB")
st.image(orig_pil, caption="Original", use_column_width=True)

with st.spinner("Enhancing with AI…"):
    out_np = enhance_image(
        pil_to_np(orig_pil),
        upscale=upscale,
        denoise_strength=denoise,
        bg_smooth_strength=bg_smooth,
        tone={"brightness": brightness, "contrast": contrast, "saturation": saturation}
    )

out_pil = Image.fromarray(out_np)
st.image(out_pil, caption="Enhanced", use_column_width=True)

# Downloads
buf_png = io.BytesIO()
out_pil.save(buf_png, format="PNG", optimize=True)
st.download_button("⬇️ Download PNG", data=buf_png.getvalue(), file_name="enhanced.png", mime="image/png")

buf_jpg = io.BytesIO()
out_pil.save(buf_jpg, format="JPEG", quality=95, optimize=True)
st.download_button("⬇️ Download JPEG (Q95)", data=buf_jpg.getvalue(), file_name="enhanced.jpg", mime="image/jpeg")
