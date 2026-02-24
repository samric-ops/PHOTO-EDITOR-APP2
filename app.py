import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io

st.set_page_config(
    page_title="Photo Enhancer",
    page_icon="🖼️"
)

st.title("🖼️ Simple Photo Enhancer")
st.write("Upload a photo to enhance it")

# Simple enhancement function using only PIL
def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    img = image.copy()
    
    # Apply enhancements
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    
    return img

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
)

if uploaded_file is not None:
    # Load image
    original_image = Image.open(uploaded_file).convert('RGB')
    
    # Show original
    st.subheader("Original Image")
    st.image(original_image, use_column_width=True)
    
    # Enhancement controls
    st.subheader("Enhancement Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    with col2:
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    with col3:
        sharpness = st.slider("Sharpness", 0.0, 5.0, 1.0, 0.1)
    
    # Add some filters
    filter_option = st.selectbox(
        "Add Filter (Optional)",
        ["None", "BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "EMBOSS", "SHARPEN", "SMOOTH"]
    )
    
    # Enhance button
    if st.button("✨ Enhance Image", type="primary"):
        with st.spinner("Enhancing..."):
            # Apply basic enhancements
            enhanced = enhance_image(original_image, brightness, contrast, sharpness)
            
            # Apply filter if selected
            if filter_option == "BLUR":
                enhanced = enhanced.filter(ImageFilter.BLUR)
            elif filter_option == "CONTOUR":
                enhanced = enhanced.filter(ImageFilter.CONTOUR)
            elif filter_option == "DETAIL":
                enhanced = enhanced.filter(ImageFilter.DETAIL)
            elif filter_option == "EDGE_ENHANCE":
                enhanced = enhanced.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_option == "EMBOSS":
                enhanced = enhanced.filter(ImageFilter.EMBOSS)
            elif filter_option == "SHARPEN":
                enhanced = enhanced.filter(ImageFilter.SHARPEN)
            elif filter_option == "SMOOTH":
                enhanced = enhanced.filter(ImageFilter.SMOOTH)
            
            # Show enhanced image
            st.subheader("Enhanced Image")
            st.image(enhanced, use_column_width=True)
            
            # Download button
            buf = io.BytesIO()
            enhanced.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Download Enhanced Image",
                data=byte_im,
                file_name="enhanced_image.png",
                mime="image/png"
            )
else:
    st.info("👆 Please upload an image to start")

# Footer
st.markdown("---")
st.caption("Made with Streamlit")
