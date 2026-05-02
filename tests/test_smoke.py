from pathlib import Path

def test_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'apps' / 'streamlit_dashboard.py').exists()
    assert (root / 'requirements.txt').exists()
    assert (root / 'notebooks' / 'HEADS_CERT_R4_2_Pipeline.ipynb').exists()
