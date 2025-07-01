
import re
import time
import numpy as np
import pandas as pd
import unicodedata

from tqdm import tqdm
from nltk.corpus import stopwords
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityPipeline:
    def __init__(
        self,
        df,
        text_column='title',
        batch_size=500,
        threshold=0.5,
        max_features=5000,
        min_df=2,
        n_components=100,
        n_results=None,
        language='portuguese'
    ):
        '''
        Pipeline para calcular similaridades entre textos usando TF-IDF + SVD.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame com a coluna de títulos.
        text_column : str
            Nome da coluna que contém o título.
        batch_size : int
            Tamanho de cada lote para cálculo de similaridade.
        threshold : float
            Limiar mínimo de similaridade para considerar par.
        max_features : int
            Número máximo de termos no TF-IDF.
        min_df : int
            Frequência mínima para incluir termo no TF-IDF.
        n_components : int
            Componentes para TruncatedSVD.
        n_results : int ou None
            Quantos pares mais similares retornar (None = todos).
        language : str
            Idioma para baixar stop-words do NLTK.
        '''
        self.df = df.copy()
        self.text_column = text_column
        self.batch_size = batch_size
        self.threshold = threshold
        self.max_features = max_features
        self.min_df = min_df
        self.n_components = n_components
        self.n_results = n_results

        # prepara stop-words
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))

        # inicializa TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df
        )
        self.tfidf_matrix = None
        self.reduced_matrix = None
        self.similarity_df = None

    def _preprocess_text(self, text, lowercase=True, remove_accents=True, remove_special_chars=False):
        '''Normaliza o texto: minusculiza, remove acentos e opcionalmente caracteres especiais.'''
        if lowercase:
            text = text.lower()
        if remove_accents:
            text = ''.join(
                c for c in unicodedata.normalize('NFKD', text)
                if not unicodedata.combining(c)
            )
        if remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def _tokenize_re(self, text):
        '''Tokeniza texto usando regex, extraindo sequências alfanuméricas.'''
        return re.findall(r'\b\w+\b', text)

    def _remove_multilang_stopwords(self, tokens):
        '''Remove stop-words pré-definidas de uma lista de tokens.'''
        return [w for w in tokens if w not in self.stop_words]

    def _prepare_texts(self):
        '''Aplica todo pré-processamento e gera coluna 'processed_text'.'''
        self.df['processed_text'] = (
            self.df[self.text_column]
                .apply(self._preprocess_text)
                .apply(self._tokenize_re)
                .apply(self._remove_multilang_stopwords)
                .apply(lambda toks: ' '.join(toks))
        )
        return self

    def _build_matrix(self):
        '''Gera matriz TF-IDF e aplica TruncatedSVD se necessário.'''
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_text'])
        if self.tfidf_matrix.shape[1] > self.n_components:
            svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            self.reduced_matrix = svd.fit_transform(self.tfidf_matrix)
        else:
            self.reduced_matrix = self.tfidf_matrix.toarray()
        return self

    def _compute_similarities(self):
        '''
        Calcula similaridades por lotes (batch) e filtra pares acima do threshold.
        Retorna DataFrame com ['title_1','title_2','score'].
        '''
        n = len(self.df)
        results = []
        for start in tqdm(range(0, n, self.batch_size), desc='Lotes'):
            end = min(start + self.batch_size, n)
            block = self.reduced_matrix[start:end]
            sims = cosine_similarity(block, self.reduced_matrix)
            rows, cols = np.where(sims > self.threshold)
            for r, c in zip(rows, cols):
                abs_r = start + r
                if c > abs_r:
                    results.append([
                        self.df.iloc[abs_r][self.text_column],
                        self.df.iloc[c][self.text_column],
                        sims[r, c]
                    ])
        sim_df = pd.DataFrame(results, columns=['title_1','title_2','score'])
        sim_df = sim_df.sort_values('score', ascending=False).reset_index(drop=True)
        if self.n_results:
            sim_df = sim_df.head(self.n_results)
        self.similarity_df = sim_df
        return self

    def run(self):
        '''Executa todo o pipeline e retorna o DataFrame de similaridades.'''
        t0 = time.time()
        (self
            ._prepare_texts()
            ._build_matrix()
            ._compute_similarities()
        )
        print(f'Pipeline executado em {time.time() - t0:.2f}s')
        return self.similarity_df

