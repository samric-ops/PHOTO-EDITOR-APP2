# AI Portrait Cleanup (Streamlit)

A simple Streamlit app that uses AI-based subject segmentation to:
- Auto-crop around the subject
- Blur or replace the background
- Adjust brightness/contrast
- Export in standard sizes (Original, 1×1, 2×2, Passport)

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
