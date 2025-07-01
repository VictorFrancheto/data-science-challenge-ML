"""
Exemplo de uso via terminal:

    python main.py
    python main.py --file produtos_amostra.csv

Este script carrega um arquivo CSV da pasta /data, executa uma pipeline de similaridade
de títulos de produtos, e salva os resultados na pasta /outputs.
"""

# main.py
import argparse
from data_loading import load_data, save_results
from similarity import SimilarityPipeline

# Argumentos via terminal
parser = argparse.ArgumentParser(description="Pipeline de similaridade de produtos")
parser.add_argument('--file', type=str, default='items_titles_test.csv', help='Nome do arquivo CSV dentro de /data')
args = parser.parse_args()

# Carregar dados
df = load_data(filename=args.file)

# Renomear coluna esperada
df = df.rename(columns={'ITE_ITEM_TITLE': 'title'})

# Executar pipeline
pipeline = SimilarityPipeline(
    df,
    text_column='title',
    batch_size=1000,
    threshold=0.5,
    max_features=5000,
    min_df=2,
    n_components=100,
    n_results=None
)

sim_df = pipeline.run()
save_path = save_results(sim_df)
print(f"Resultados salvos em: {save_path}")
