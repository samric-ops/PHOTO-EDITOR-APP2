import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np

# --- Optional: show env versions early in logs (helpful during first deploy) ---
import sys
print("Python:", sys.version)
try:
    import numpy as _np
    print("numpy:", _np.__version__)
    import cv2 as _cv2
    print("cv2:", _cv2.__version__)
except Exception as _e:
    print("Early import check failed:", repr(_e))
    # Don't raise here; Streamlit Cloud sometimes imports twice. Real import happens later.


# ===================== Utility functions =====================

def lazy_cv2():
    """Import cv2 only when needed to avoid startup glitches."""
    import cv2
    return cv2

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    cv2 = lazy_cv2()
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img: np.ndarray) -> Image.Image:
    cv2 = lazy_cv2()
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def apply_tone(img_bgr: np.ndarray, brightness=0, contrast=0, saturation=0):
    cv2 = lazy_cv2()
    # brightness/contrast
    c = np.clip(contrast, -100, 100)
    b = np.clip(brightness, -100, 100)
    alpha = 1 + (c / 100.0)
    beta = b
    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    # saturation
    if saturation != 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[...,1] = np.clip(hsv[...,1] * (1 + saturation/100.0), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def smooth_background(img_bgr: np.ndarray, strength=10):
    """Light denoise that preserves edges to keep the overall image clean."""
    cv2 = lazy_cv2()
    s = max(5, int(strength))
    return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=s*3, sigmaSpace=s)


# ===================== Load Models (cached) =====================

@st.cache_resource(show_spinner="Loading enhancement models…")
def load_models():
    """
    Loads GFPGAN + Real-ESRGAN. We use Real-ESRGAN x4 as the background upsampler.
    """
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer

    cache_dir = Path.home() / ".cache" / "ai-photo-enhancer"
    os.makedirs(cache_dir, exist_ok=True)

    # Real-ESRGAN model (x4plus) – great all-around super-resolution
    sr_upsampler = RealESRGANer(
        scale=4,
        model_path=None,              # let library auto-download
        model="RealESRGAN_x4plus",
        tile=200, tile_pad=10, pre_pad=0,
        half=True                     # FP16 if supported
    )

    # GFPGAN face restoration, using the same SR for background
    face_enhancer = GFPGANer(
        model_path=None,              # auto-download
        upscale=4,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=sr_upsampler
    )
    return face_enhancer, sr_upsampler


def enhance_image(
    img_bgr: np.ndarray,
    upscale: int = 2,
    sr_denoise_strength: float = 0.5,
    bg_smooth_strength: int = 10,
    tone: dict | None = None
) -> np.ndarray:
    """
    End-to-end enhancement: denoise -> restore -> upscale -> smooth -> tone.
    """
    cv2 = lazy_cv2()
    face_enhancer, sr_upsampler = load_models()

    # Pre-denoise to help the SR/restoration produce cleaner textures
    if sr_denoise_strength > 0:
        h, w = img_bgr.shape[:2]
        # Non-local means denoising (conservative)
        img_bgr = cv2.fastNlMeansDenoisingColored(
            img_bgr, None,
            5 + int(sr_denoise_strength * 5),
            5 + int(sr_denoise_strength * 5),
            7, 21
        )

    # Use GFPGAN pipeline (restores faces + upscales background via Real-ESRGAN)
    _, _, restored = face_enhancer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    out = restored

    # Adjust final size depending on desired upscale factor
    # GFPGAN returns ~4x vs input; downscale to 2x if user selected 2
    if upscale == 2:
        rh, rw = out.shape[:2]
        out = cv2.resize(out, (rw // 2, rh // 2), interpolation=cv2.INTER_CUBIC)

    # Light background smoothing for cleaner overall appearance
    if bg_smooth_strength > 0:
        out = smooth_background(out, strength=bg_smooth_strength)

    # Tone adjustments
    if tone:
        out = apply_tone(
            out,
            brightness=tone.get("brightness", 0),
            contrast=tone.get("contrast", 8),
            saturation=tone.get("saturation", 6)
        )

    return out


# ===================== Streamlit UI =====================

st.set_page_config(page_title="AI Photo Enhancer", page_icon="✨", layout="centered")
st.title("✨ AI Photo Enhancer")
st.caption("Upscale • Denoise • Restore • Natural tones")

with st.sidebar:
    st.header("Quality Controls")
    upscale = st.selectbox("Upscale factor", [2, 4], index=0)  # 2x is often enough for print/ID use
    sr_denoise = st.slider("Denoise strength", 0.0, 1.0, 0.5, 0.05)
    bg_smooth = st.slider("Background smoothing", 0, 30, 10)
    st.subheader("Tone")
    brightness = st.slider("Brightness", -50, 50, 0)
    contrast = st.slider("Contrast", -50, 50, 8)
    saturation = st.slider("Saturation", -30, 30, 6)
    st.caption("Tip: Small positive contrast and saturation add clean, natural pop.")

uploaded = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if not uploaded:
    st.info("Upload an image to start.")
    st.stop()

# Preview original
orig_pil = Image.open(uploaded).convert("RGB")
st.image(orig_pil, caption="Original", use_column_width=True)

# Enhance
with st.spinner("Enhancing with AI…"):
    out_bgr = enhance_image(
        pil_to_bgr(orig_pil),
        upscale=upscale,
        sr_denoise_strength=sr_denoise,
        bg_smooth_strength=bg_smooth,
        tone={"brightness": brightness, "contrast": contrast, "saturation": saturation}
    )

out_pil = bgr_to_pil(out_bgr)
st.image(out_pil, caption="Enhanced", use_column_width=True)

# Downloads
png_buf = io.BytesIO()
out_pil.save(png_buf, format="PNG", optimize=True)
st.download_button("⬇️ Download PNG", data=png_buf.getvalue(), file_name="enhanced.png", mime="image/png")

jpg_buf = io.BytesIO()
out_pil.save(jpg_buf, format="JPEG", quality=95, optimize=True)
st.download_button("⬇️ Download JPEG (Q95)", data=jpg_buf.getvalue(), file_name="enhanced.jpg", mime="image/jpeg")
