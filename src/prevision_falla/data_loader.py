import pandas as pd
from pathlib import Path
from config import DATA_DIR, RESULTS_DIR

def load_data(filename='full_devices.csv'):
    '''
    Carga un archivo CSV desde el directorio de datos.

    Parámetros:
    ----------
    filename : str, opcional
        Nombre del archivo CSV a cargar. Por defecto es 'full_devices.csv'.

    Retorna:
    -------
    pd.DataFrame
        Un DataFrame con los datos cargados desde el archivo.

    Lanza:
    -----
    FileNotFoundError
        Si el archivo especificado no se encuentra en el directorio.
    '''
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    df = pd.read_csv(filepath, encoding='latin1')

    return df


def save_predictions(df, filename='predictions.csv'):
    '''
    Guarda un DataFrame con las columnas ['prob_failure', 'predicted_class', 'true_class']
    en el directorio /results.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene dichas columnas.
        filename (str): Nombre del archivo de salida.
    '''
    output_dir = DATA_DIR / filename
    output_dir = Path(__file__).resolve().parent.parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_df = df[['prob_failure', 'predicted_class', 'true_class']]
    output_path = output_dir / filename
    output_df.to_csv(output_path, index=False)
    print(f'Resultados salvos em: {output_path}')


if __name__ == '__main__':
    df = load_data()
    print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
