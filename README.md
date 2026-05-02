# DEMF GUI Training Pack

Files included:
- app_streamlit.py
- demf_core.py
- config.yaml
- requirements.txt

## Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## What is fixed
- safe charts when report files are empty or missing columns
- no crash when `final_score` is not present yet
- training controls moved into the GUI
- raw CSV upload and save from the GUI
- proper evaluation display using the current `evaluation_summary.json` structure
- 80/20 time-based train/test split stored in `meta.json`
