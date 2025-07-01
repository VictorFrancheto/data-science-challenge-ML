from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # seu_projeto/
ROOT_DIR = BASE_DIR.parent                           # projeto/
DATA_DIR = ROOT_DIR / 'data'                         # projeto/data
RESULTS_DIR = ROOT_DIR / 'results'  