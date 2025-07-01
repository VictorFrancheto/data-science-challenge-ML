import pandas as pd
from pathlib import Path
from config import DATA_DIR, RESULTS_DIR

def load_data(filename='full_devices.csv'):
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    df = pd.read_csv(filepath, encoding='latin1')
    return df

from pathlib import Path

def save_predictions(df, filename='predictions.csv'):
    '''
    Salva um DataFrame com as colunas ['prob_failure', 'predicted_class', 'true_class']
    no diretório /results.

    Args:
        df (pd.DataFrame): DataFrame contendo essas colunas.
        filename (str): nome do arquivo de saída.
    '''
    output_dir = DATA_DIR / filename
    output_dir = Path(__file__).resolve().parent.parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_df = df[['prob_failure', 'predicted_class', 'true_class']]
    output_path = output_dir / filename
    output_df.to_csv(output_path, index=False)
    print(f'Resultados salvos em: {output_path}')


if __name__ == "__main__":
    df = load_data()
    print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
