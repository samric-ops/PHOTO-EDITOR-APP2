import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io

st.set_page_config(
    page_title="Simple Photo Enhancer",
    page_icon="📸",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        padding: 10px;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #00FF0011;
        border: 1px solid #00FF00;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📸 Simple Photo Enhancer")
st.markdown("Enhance your photos with easy-to-use filters")

# Initialize session state
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Enhancement Controls")
    st.markdown("---")
    
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    sharpness = st.slider("Sharpness", 0.0, 5.0, 1.0, 0.1)
    color = st.slider("Color Saturation", 0.0, 2.0, 1.0, 0.1)
    
    st.markdown("---")
    st.header("🔄 Filters")
    
    apply_sharpen = st.checkbox("Sharpen Filter")
    apply_smooth = st.checkbox("Smooth Filter")
    apply_edge = st.checkbox("Edge Enhance")
    
    st.markdown("---")
    st.info("💡 Tip: Adjust sliders to see real-time preview")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload your image here"
    )

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display original
    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_columnwidth=True)
        st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
    
    # Process image
    with col2:
        st.subheader("✨ Enhanced Image")
        
        # Apply enhancements
        enhanced = image.copy()
        
        # Basic enhancements
        enhanced = ImageEnhance.Brightness(enhanced).enhance(brightness)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
        enhanced = ImageEnhance.Color(enhanced).enhance(color)
        
        # Apply filters
        if apply_sharpen:
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
        if apply_smooth:
            enhanced = enhanced.filter(ImageFilter.SMOOTH)
        if apply_edge:
            enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE)
        
        # Store in session state
        st.session_state.enhanced_image = enhanced
        
        # Display enhanced image
        st.image(enhanced, use_columnwidth=True)
        
        # Download button
        buf = io.BytesIO()
        enhanced.save(buf, format='PNG')
        byte_im = buf.getvalue()
        
        st.download_button(
            label="📥 Download Enhanced Image",
            data=byte_im,
            file_name="enhanced_image.png",
            mime="image/png",
            use_container_width=True
        )
    
    # Success message
    st.markdown("""
        <div class="success-box">
            ✅ Image successfully enhanced! Use the download button to save it.
        </div>
    """, unsafe_allow_html=True)

else:
    # Welcome message
    with col2:
        st.info("👈 Upload an image to start enhancing!")
        st.markdown("""
        ### Features:
        - ✨ Real-time preview
        - 🎛️ Adjustable brightness, contrast, sharpness, and color
        - 🔄 Multiple filter options
        - 📥 Easy download
        """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
