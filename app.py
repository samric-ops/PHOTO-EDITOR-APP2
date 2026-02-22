import io
import zipfile
from datetime import datetime

import numpy as np
from PIL import Image
import cv2
import streamlit as st

# MediaPipe for AI subject segmentation
import mediapipe as mp

st.set_page_config(page_title="AI Portrait Cleanup", page_icon="📸", layout="centered")

# ---------- Utilities ----------
@st.cache_resource
def load_segmenter():
    # model_selection=1 is better for portrait-like images
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def get_mask_bgr(img_bgr: np.ndarray, segmenter) -> np.ndarray:
    """Return mask (H, W) float32 in [0,1] where 1 indicates subject."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = segmenter.process(img_rgb)
    mask = result.segmentation_mask.astype("float32")
    # Normalize robustly to [0,1]
    mask = np.clip(mask, 0.0, 1.0)
    return mask

def auto_crop(img_bgr: np.ndarray, mask: np.ndarray, padding_ratio=0.15) -> np.ndarray:
    """Crop tightly around the subject with padding_ratio (0–0.5)."""
    h, w = mask.shape
    thresh = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr  # fallback: no crop
    x, y, w_box, h_box = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Add padding
    pad_x = int(w_box * padding_ratio)
    pad_y = int(h_box * padding_ratio)
    x0 = max(x - pad_x, 0)
    y0 = max(y - pad_y, 0)
    x1 = min(x + w_box + pad_x, w)
    y1 = min(y + h_box + pad_y, h)

    return img_bgr[y0:y1, x0:x1]

def blur_background(img_bgr: np.ndarray, mask: np.ndarray, ksize=35) -> np.ndarray:
    """Apply Gaussian blur to background only."""
    k = max(3, int(ksize) | 1)  # kernel must be odd
    blurred = cv2.GaussianBlur(img_bgr, (k, k), 0)
    mask_3 = np.dstack([mask]*3)
    out = (img_bgr * mask_3 + blurred * (1 - mask_3)).astype(np.uint8)
    return out

def replace_background_color(img_bgr: np.ndarray, mask: np.ndarray, rgb_color=(255,255,255)) -> np.ndarray:
    bg = np.zeros_like(img_bgr, dtype=np.uint8)
    bg[:, :] = (rgb_color[2], rgb_color[1], rgb_color[0])  # RGB->BGR
    mask_3 = np.dstack([mask]*3)
    out = (img_bgr * mask_3 + bg * (1 - mask_3)).astype(np.uint8)
    return out

def adjust_brightness_contrast(img_bgr: np.ndarray, brightness=0, contrast=0):
    """
    brightness: -100..100, contrast: -100..100
    """
    b = np.clip(brightness, -100, 100)
    c = np.clip(contrast, -100, 100)

    if c >= 0:
        alpha = 1 + (c / 100) * 2.0  # up to 3.0
    else:
        alpha = 1 + (c / 100)        # down to 0.0
    beta = b  # -100..100

    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return out

def resize_for_export(img_bgr: np.ndarray, target):
    if target == "Original":
        return img_bgr
    sizes = {
        "1×1 (300×300)": (300, 300),
        "2×2 (600×600)": (600, 600),
        "Passport (600×600)": (600, 600),
    }
    if target in sizes:
        return cv2.resize(img_bgr, sizes[target], interpolation=cv2.INTER_CUBIC)
    return img_bgr

# ---------- UI ----------
st.title("📸 AI Portrait Cleanup")
st.caption("Upload photos, then blur or replace backgrounds, auto-crop, and export in standard sizes.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Background mode", ["Blur background", "Solid color background"])
    if mode == "Blur background":
        blur_strength = st.slider("Blur strength", min_value=5, max_value=81, value=35, step=2)
    else:
        pick_color = st.color_picker("Background color", "#FFFFFF")
        # Convert hex to RGB tuple
        bg_rgb = tuple(int(pick_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    padding_ratio = st.slider("Auto-crop padding (%)", 0, 50, 15) / 100.0
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", -100, 100, 0)
    export_size = st.selectbox("Export size", ["Original", "1×1 (300×300)", "2×2 (600×600)", "Passport (600×600)"])

uploaded = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded:
    segmenter = load_segmenter()
    outputs = []

    for file in uploaded:
        # Load image
        img_pil = Image.open(file).convert("RGB")
        img_bgr = pil_to_cv2(img_pil)

        # Get mask and crop
        mask = get_mask_bgr(img_bgr, segmenter)
        img_cropped = auto_crop(img_bgr, mask, padding_ratio=padding_ratio)
        # Recompute mask for the cropped image for best results
        mask_cropped = get_mask_bgr(img_cropped, segmenter)

        # Background processing
        if mode == "Blur background":
            processed = blur_background(img_cropped, mask_cropped, ksize=blur_strength)
        else:
            processed = replace_background_color(img_cropped, mask_cropped, rgb_color=bg_rgb)

        # Adjustments
        processed = adjust_brightness_contrast(processed, brightness=brightness, contrast=contrast)

        # Final resize
        exported = resize_for_export(processed, export_size)

        # Show preview
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2_to_pil(img_bgr), caption=f"Original: {file.name}", use_column_width=True)
        with col2:
            st.image(cv2_to_pil(exported), caption=f"Processed: {file.name}", use_column_width=True)

        # Per-file download
        out_name = f"{file.name.rsplit('.', 1)[0]}_processed.png"
        buf = io.BytesIO()
        cv2_to_pil(exported).save(buf, format="PNG", optimize=True)
        st.download_button("⬇️ Download processed image", data=buf.getvalue(), file_name=out_name, mime="image/png")

        outputs.append((out_name, buf.getvalue()))

    # Batch zip download
    if len(outputs) > 1:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in outputs:
                zf.writestr(name, data)
        st.download_button(
            "⬇️ Download all as ZIP",
            data=zbuf.getvalue(),
            file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )
else:
    st.info("Upload one or more images to begin.")
