# DEMF GUI Training Pack

Files included:
- app_streamlit.py
- demf_core.py
- config.yaml
- requirements.txt

## Run

### Option 1: PowerShell helper

```powershell
.
un_project.ps1
```

### Option 2: manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app_streamlit.py
```
.\run_project.ps1
> Note: `app_streamlit.py` now forwards to `app.py`, which contains the full multi-product dashboard with BCAS, NSADM, HEADS, and DEMF.

## What is fixed
- safe charts when report files are empty or missing columns
- no crash when `final_score` is not present yet
- training controls moved into the GUI
- raw CSV upload and save from the GUI
- proper evaluation display using the current `evaluation_summary.json` structure
- 80/20 time-based train/test split stored in `meta.json`
