from pathlib import Path
import datetime

BASE_DIR = Path(__file__).resolve().parent.parent          # seu_projeto/
ROOT_DIR = BASE_DIR.parent                           # projeto/
DATA_DIR = ROOT_DIR / 'data'                         # projeto/data
RESULTS_DIR = ROOT_DIR / 'results'                   # opcional, se quiser salvar fora também


# Configurações aumentadas
PREPROCESSING_CONFIG = {
    'lowercase': True,
    'remove_accents': True,
    'remove_special_chars': True,
    'min_word_length': 3,
    'chunk_size': 10000  # Novo parâmetro
}

MODEL_CONFIG = {
    'batch_size': 1000,
    'threshold': 0.5,
    'max_features': 5000,
    'min_df': 2,
    'n_components': 100,
    'language': 'portuguese',
    'min_memory': 1 * 1024**3,  # 1GB mínimo requerido
    'n_estimators': 20,         # Para LSH Forest
    'n_candidates': 100         # Para LSH Forest
}

PLOT_CONFIG = {
    'color_primary': '#FFE600',
    'color_secondary': '#2D3277',
    'bg_color': 'white'
}