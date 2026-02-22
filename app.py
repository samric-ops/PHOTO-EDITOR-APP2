import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np

# ============================================================
# WALANG top-level `import cv2` dito. Gumamit tayo ng lazy import.
# ============================================================

def lazy_cv2():
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
    c = np.clip(contrast, -100, 100)
    b = np.clip(brightness, -100, 100)
    alpha = 1 + (c / 100.0)
    beta = b
    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    if saturation != 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + saturation / 100.0), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def smooth_background(img_bgr: np.ndarray, strength=10):
    cv2 = lazy_cv2()
    s = max(5, int(strength))
    return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=s*3, sigmaSpace=s)

@st.cache_resource(show_spinner="Loading enhancement models…")
def load_models():
    # I‑import ang heavy deps sa loob para iwas early import failures
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer

    cache_dir = Path.home() / ".cache" / "ai-photo-enhancer"
    os.makedirs(cache_dir, exist_ok=True)

    sr_upsampler = RealESRGANer(
        scale=4,
        model_path=None,      # auto-download
        model="RealESRGAN_x4plus",
        tile=200, tile_pad=10, pre_pad=0,
        half=True
    )

    face_enhancer = GFPGANer(
        model_path=None,      # auto-download
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
    cv2 = lazy_cv2()
    face_enhancer, _ = load_models()

    # Conservative pre-denoise
    if sr_denoise_strength > 0:
        img_bgr = cv2.fastNlMeansDenoisingColored(
            img_bgr, None,
            5 + int(sr_denoise_strength * 5),
            5 + int(sr_denoise_strength * 5),
            7, 21
        )

    # Face restoration + SR (background upscaled by Real-ESRGAN)
    _, _, restored = face_enhancer.enhance(
        img_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )
    out = restored

    # Downscale to 2× if user selected 2× (GFPGAN returns ~4×)
    if upscale == 2:
        h, w = out.shape[:2]
        out = cv2.resize(out, (w // 2, h // 2), interpolation=cv2.INTER_CUBIC)

    # Light background smoothing
    if bg_smooth_strength > 0:
        out = smooth_background(out, strength=bg_smooth_strength)

    # Tone
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
    upscale = st.selectbox("Upscale factor", [2, 4], index=0)
    sr_denoise = st.slider("Denoise strength", 0.0, 1.0, 0.5, 0.05)
    bg_smooth = st.slider("Background smoothing", 0, 30, 10)
    st.subheader("Tone")
    brightness = st.slider("Brightness", -50, 50, 0)
    contrast   = st.slider("Contrast",   -50, 50, 8)
    saturation = st.slider("Saturation", -30, 30, 6)
    st.caption("Tip: Small positive contrast and saturation add clean, natural pop.")

uploaded = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if not uploaded:
    st.info("Upload an image to start.")
    st.stop()

orig_pil = Image.open(uploaded).convert("RGB")
st.image(orig_pil, caption="Original", use_column_width=True)

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
buf_png = io.BytesIO()
out_pil.save(buf_png, format="PNG", optimize=True)
st.download_button("⬇️ Download PNG", data=buf_png.getvalue(), file_name="enhanced.png", mime="image/png")

buf_jpg = io.BytesIO()
out_pil.save(buf_jpg, format="JPEG", quality=95, optimize=True)
st.download_button("⬇️ Download JPEG (Q95)", data=buf_jpg.getvalue(), file_name="enhanced.jpg", mime="image/jpeg")
