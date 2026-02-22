# AI Photo Enhancer (Streamlit)

## Local run
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

## Deploy to Streamlit Cloud
- Push to GitHub
- New app → select repo → main file: `app.py`
- First run may take longer (models auto-download)
