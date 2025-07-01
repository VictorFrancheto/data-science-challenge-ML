import pytest
import pandas as pd
from similitud_productos.similarity import SimilarityPipeline  # ajuste para o caminho correto

# Dados de exemplo para teste
data = {
    'title': [
        "Camiseta azul tamanho M",
        "Camiseta azul tamanho G",
        "Calça jeans azul",
        "Sapato preto tamanho 42",
        "Tênis branco esportivo"
    ]
}

df = pd.DataFrame(data)

def test_preprocess_text_basic():
    pipeline = SimilarityPipeline(df)
    texto = "Camiseta ÁZUL!!!"
    resultado = pipeline._preprocess_text(texto)
    assert resultado == "camiseta azul!!!".lower()  # sem remoção de caracteres especiais

def test_preprocess_text_remove_accents_and_special():
    pipeline = SimilarityPipeline(df)
    texto = "Camiseta ÁZUL!!!"
    resultado = pipeline._preprocess_text(texto, remove_special_chars=True)
    assert resultado == "camiseta azul"

def test_tokenize_re():
    pipeline = SimilarityPipeline(df)
    texto = "Camiseta azul tamanho M"
    tokens = pipeline._tokenize_re(texto)
    assert tokens == ["Camiseta", "azul", "tamanho", "M"]

def test_remove_multilang_stopwords():
    pipeline = SimilarityPipeline(df)
    tokens = ["a", "camiseta", "é", "azul"]
    filtrados = pipeline._remove_multilang_stopwords(tokens)
    # "a" e "é" são stopwords em português e devem ser removidos
    assert "a" not in filtrados
    assert "é" not in filtrados
    assert "camiseta" in filtrados

def test_pipeline_run():
    pipeline = SimilarityPipeline(df, batch_size=2, threshold=0.1, n_results=5)
    sim_df = pipeline.run()
    assert isinstance(sim_df, pd.DataFrame)
    assert set(['title_1', 'title_2', 'score']).issubset(sim_df.columns)
    # Deve haver pelo menos 1 par similar
    assert len(sim_df) > 0

    # Checa se scores estão acima do threshold
    assert (sim_df['score'] >= 0.1).all()

if __name__ == "__main__":
    pytest.main()
