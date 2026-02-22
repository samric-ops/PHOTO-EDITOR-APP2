import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# --- Streamlit page ---
st.set_page_config(page_title="AI Photo Enhancer", page_icon="✨", layout="centered")
st.title("✨ AI Photo Enhancer (GFPGAN + Real-ESRGAN)")
st.caption("Upscale • Denoise • Sharpen • Natural skin tones")

# ============ Utilities ============
def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def apply_tone_adjustments(img_bgr: np.ndarray, contrast=0, brightness=0, saturation=0):
    # contrast/brightness via OpenCV
    c = np.clip(contrast, -100, 100)
    b = np.clip(brightness, -100, 100)
    alpha = 1 + (c / 100.0)  # 0..2
    beta = b                 # -100..100
    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    # saturation in HSV
    if saturation != 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[...,1] = np.clip(hsv[...,1] * (1 + saturation/100.0), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def smooth_background(img_bgr: np.ndarray, strength=15):
    # Bilateral smooth—keeps edges crisp while smoothing noise
    s = max(5, int(strength))
    return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=s*3, sigmaSpace=s)

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    # Lazy import heavy deps only once
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer

    model_dir = str(Path.home() / ".cache" / "ai-photo-enhancer")
    os.makedirs(model_dir, exist_ok=True)

    # Real-ESRGAN model (x4plus) – good general purpose
    sr_model = RealESRGANer(
        scale=4,
        model_path=None,  # auto-download
        model="RealESRGAN_x4plus",
        tile=200, tile_pad=10, pre_pad=0,
        half=True  # use half precision if supported
    )

    # GFPGAN for face restoration
    face_enhancer = GFPGANer(
        model_path=None,  # auto-download
        upscale=4,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=sr_model  # use SR model for background
    )
    return face_enhancer, sr_model

def enhance_image(
    img_bgr: np.ndarray,
    upscale: int = 4,
    face_strength: float = 1.0,
    sr_denoise_strength: float = 0.5,
    tone_cfg: dict = None,
    bg_smooth: int = 10
) -> np.ndarray:
    face_enhancer, sr_model = load_models()

    # --- Step 1: Base super-resolution for background ---
    # RealESRGANer allows setting denoise strength by choosing model; we simulate with mild bilateral then SR
    base_in = img_bgr.copy()
    if sr_denoise_strength > 0:
        base_in = cv2.fastNlMeansDenoisingColored(base_in, None, 5+int(sr_denoise_strength*5), 5+int(sr_denoise_strength*5), 7, 21)

    # Use GFPGAN’s built-in pipeline that restores faces and upscales background via the attached SR upsampler.
    # It returns (cropped_faces, restored_faces, restored_img)
    _, _, restored = face_enhancer.enhance(
        base_in,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    out = restored

    # Optional extra upscale to match user-selected factor (if 2x only needed, downscale later)
    if upscale == 2:
        # Downscale from x4 to x2 for crisper result without being too large
        h, w = out.shape[:2]
        out = cv2.resize(out, (w//2, h//2), interpolation=cv2.INTER_CUBIC)

    # --- Step 2: Background smoothing (light) ---
    if bg_smooth > 0:
        out = smooth_background(out, strength=bg_smooth)

    # --- Step 3: Tone adjustments ---
    if tone_cfg:
        out = apply_tone_adjustments(
            out,
            contrast=tone_cfg.get("contrast", 5),
            brightness=tone_cfg.get("brightness", 0),
            saturation=tone_cfg.get("saturation", 5)
        )

    return out

# ============ UI Controls ============
with st.sidebar:
    st.header("Quality Controls")
    upscale = st.selectbox("Upscale factor", [2, 4], index=0)  # 2x is usually enough for prints
    sr_denoise = st.slider("De-noise strength (SR)", 0.0, 1.0, 0.5, 0.05)
    bg_smooth = st.slider("Background smoothing", 0, 30, 10)
    st.divider()
    st.subheader("Tone")
    brightness = st.slider("Brightness", -50, 50, 0)
    contrast = st.slider("Contrast", -50, 50, 8)
    saturation = st.slider("Saturation", -30, 30, 6)
    st.caption("Tip: Small positive contrast and saturation give a clean, natural look.")

uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if not uploaded:
    st.info("Mag‑upload ng larawan para simulan ang enhancement.")
    st.stop()

image_pil = Image.open(uploaded).convert("RGB")
st.image(image_pil, caption="Original", use_column_width=True)

with st.spinner("Enhancing with AI..."):
    out_bgr = enhance_image(
        pil_to_bgr(image_pil),
        upscale=upscale,
        sr_denoise_strength=sr_denoise,
        tone_cfg={"brightness": brightness, "contrast": contrast, "saturation": saturation},
        bg_smooth=bg_smooth
    )

out_pil = bgr_to_pil(out_bgr)
st.image(out_pil, caption="Enhanced", use_column_width=True)

# Downloads
buf_png = io.BytesIO()
out_pil.save(buf_png, format="PNG", optimize=True)
st.download_button("⬇️ Download PNG", data=buf_png.getvalue(), file_name="enhanced.png", mime="image/png")

buf_jpg = io.BytesIO()
out_pil.save(buf_jpg, format="JPEG", quality=95, optimize=True)
st.download_button("⬇️ Download JPEG (Q95)", data=buf_jpg.getvalue(), file_name="enhanced.jpg", mime="image/jpeg")
