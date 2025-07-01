import pandas as pd
from config import DATA_DIR, RESULTS_DIR
import numpy as np

def load_data(filename='items_titles_test.csv', chunksize=None):
    """Carrega dados em chunks se especificado"""
    filepath = DATA_DIR / filename
    df = pd.read_csv(filepath)
    return df

def save_results(df, filename='similarity_results.csv'):
    """Salva os resultados em partes se for muito grande"""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(df) > 1_000_000:  # Se maior que 1M linhas
        for i, chunk in enumerate(np.array_split(df, 10)):
            chunk.to_csv(output_path.with_stem(f"{output_path.stem}_part{i+1}"), 
                        index=False, sep=';')
    else:
        df.to_csv(output_path, index=False, sep=';')
    
    return output_path
