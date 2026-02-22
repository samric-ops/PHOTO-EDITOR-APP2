# AI Photo Enhancer (Streamlit)

## Local run
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

## Deploy to Streamlit Community Cloud
- Push to GitHub
- New app → select repo → main file: `app.py`
- After first build, if imports fail, use **Manage app → Reboot**.
- If necessary: **Settings → Advanced → Clear cache**, then Reboot.
