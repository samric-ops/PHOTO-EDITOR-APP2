# AI Photo Enhancer (Streamlit)

A Streamlit app that performs high-quality enhancement:
- Super-resolution (2× / 4×)
- Denoising & restoration
- Optional background smoothing
- Tone controls (brightness, contrast, saturation)
- One-click PNG/JPEG download

## Run locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
