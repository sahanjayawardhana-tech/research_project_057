# Run the merged SOC dashboard from PowerShell.
# This script creates the virtual environment if needed,
# installs dependencies, activates the venv, and starts Streamlit.

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

if (-not (Test-Path -Path .\.venv\Scripts\Activate.ps1)) {
    python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app_streamlit.py
