# Data Science Challenge: Mercado Libre


<p align="center">
  <img src="https://github.com/VictorFrancheto/data-science-challenge-ML/blob/main/image.png">
</p>


📂 **Todos los notebooks entregables se encuentran en la carpeta [`notebooks/`](./notebooks/)**.

Este repositorio contiene las soluciones a los tres ejercicios propuestos en el desafío técnico de Data Science de Mercado Libre. Cada notebook es independiente y responde a un ejercicio específico, abarcando desde análisis exploratorio hasta modelado predictivo con series temporales.

## 🧱 Estructura de Carpetas
```text
├── data/ # Archivos de entrada (.csv)
├── notebooks/ # Notebooks entregables (uno por ejercicio)
│ ├── 1_ofertas_relampago.ipynb
│ ├── 2_similitud_productos.ipynb
│ └── 3_prediccion_falla.ipynb
├── results/ # Resultados generados para los entregables
├── src/ # Funciones reutilizables y utilitarios
├── tests/ # Funciones para tests
├── README.md # Este archivo
└── pyproject.toml # Dependencias del proyecto
```
## 📌 Descripción de los Ejercicios

### 1. 🔍 Análisis de Ofertas Relámpago

**Archivo de entrada**: `ofertas_relampago.csv`  
El objetivo es realizar un análisis exploratorio (EDA) y extraer insights relevantes sobre el comportamiento de este tipo de ofertas.  
Notebook correspondiente: **notebooks/01_ofertas_relampago.ipynb**

---

### 2. 🤝 Similitud entre Productos

**Archivos de entrada**: `items_titles.csv`, `items_titles_test.csv`  
Este ejercicio consiste en calcular la similitud entre títulos de productos y listar los pares más similares, sin utilizar modelos preentrenados. También se analiza la escalabilidad y el tiempo de ejecución de la solución.  
Notebook correspondiente: **notebooks/02_similitud_products.ipynb**


Entregables: **results/similarity_results.csv**

---

### 3. ⚙️ Predicción de Fallas en Dispositivos

**Archivo de entrada**: `full_devices.csv`  
El objetivo es predecir la probabilidad de falla de un dispositivo con un día de anticipación, utilizando técnicas de series temporales y mantenimiento predictivo.  
Notebook correspondiente: **notebooks/03_previsao_falla.ipynb**

Entregables: **results/predictions.csv**

---
