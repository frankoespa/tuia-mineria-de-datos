# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Reglas del proyecto (definidas por los autores)

Trabajo practico para la materia de mineria de datos.
Se utiliza como fuente de datos el archivo `tp1/penguins_size.csv` y el desarrollo del tp 1 estará en el archivo `tp1/tp1_mineria_renna_esparza.ipynb`.
Se deberá realizar siguiendo estrictamente las librerias vistas en clase (todo lo que vimos se encuentra en la carpeta `tp1/teoria`), métodos utilizados con sus respectivos parametros. No agregar ni modificar algun parámetro extra sin explicar el por que.

El enunciado del trabajo práctico es el archivo `tp1/TP1.pdf`.

Estos trabajos prácticos son para fines educativos, explicar cada detalle para su perfecto entendimiento.

Este proyecto tiene un repositorio remoto llamado https://github.com/frankoespa/tuia-mineria-de-datos.git

## Estructura

- `tp1/TP1.pdf`: enunciado. Seis actividades: (1) EDA + limpieza + estandarización, con `Especie` como variable objetivo; (2) PCA con los 3 criterios; (3) Isomap variando vecinos y componentes; (4) t-SNE variando iteraciones, componentes y perplejidad; (5) K-means con GAP + Silhouette y un gráfico 3D; (6) clustering jerárquico con Silhouette + GAP.
- `tp1/teoria/`: notebooks de clase, que son la **referencia de qué librerías, métodos y parámetros se pueden usar**. Antes de escribir una celda, buscar cómo se hizo ahí:
  - `U1_Introducción.ipynb`: EDA, `SimpleImputer(strategy='mean')`, `Winsorizer(capping_method='iqr', tail='both', fold=1.5)` (feature-engine), `StandardScaler`, `train_test_split`.
  - `U2_Reducción_de_la_dimensionalidad.ipynb`: `Pipeline([('scaler', StandardScaler()), ('pca', PCA())])`, `Isomap(n_neighbors=..., n_components=...)`, `TSNE(n_components=..., init='random', random_state=42, method='exact')`, UMAP y MDS. Los 3 criterios para elegir componentes de PCA son: varianza acumulada (~75–80%), Kaiser (eigenvalues > 1) y codo/scree. El gráfico de varianza explicada (barras) + acumulada (línea) está hecho con matplotlib.
  - `U3_Clustering_Wheat.ipynb`: `KMeans(n_clusters=k, random_state=42)`, `AgglomerativeClustering(n_clusters=k, linkage='ward')`, `scipy.cluster.hierarchy.linkage` + `dendrogram`, `silhouette_score` / `silhouette_samples` (k de 2 a `max_k`), DBSCAN y HDBSCAN. **GAP no viene de ninguna librería**: se programa a mano con `calculate_intra_cluster_dispersion(X, k)`, que compara `log(inercia de referencia) - log(inercia real)` usando 10 conjuntos `np.random.rand(*X.shape)`, con `max_k = 10` y `optimal_k = np.argmax(gaps) + 1`. Hay una versión para K-means y otra para jerárquico. Reutilizar ese mismo enfoque.
- `tp1/tp1_mineria_renna_esparza.ipynb`: entrega. Arranca vacío y usa el kernel del venv `.entorno`.

## Particularidades del dataset (`penguins_size.csv`)

No es el CSV original de Kaggle; está modificado:
- Separador `;`: hay que leerlo con `pd.read_csv(..., sep=';')`.
- Columnas en español: `Especie`, `Longitud Culmen (mm)`, `Profundidad Culmen (mm)`, `Longitud Aleta (mm)`, `Masa corporal (g)`, `Sexo`. No existe la columna de isla.
- Tiene dos columnas finales sin nombre y totalmente vacías (vienen del `;;` al final de cada línea), que hay que eliminar.
- 347 filas. Las 4 numéricas tienen 2 faltantes cada una (filas completamente vacías salvo la especie). `Sexo` tiene 10 vacíos y 1 valor inválido `'.'`.
- Especies: `Adelie Penguin` (154), `Chinstrap penguin` (68) y `Gentoo penguin` (125). Ojo: las mayúsculas no son consistentes.

## Entorno

El venv es `.entorno` (Python 3.13, Windows). Las dependencias del TP1 están fijadas con versión en `requirements.txt`:

```powershell
.entorno\Scripts\python.exe -m pip install -r requirements.txt
```

Si se agrega una librería, fijar su versión en `requirements.txt`. `umap-learn`, `hdbscan` y `kagglehub` se usan en los notebooks de teoría pero quedaron afuera porque el TP1 no los necesita.

## Requisitos de formato de la entrega (del enunciado)

- Cabecera con año, materia e integrantes (Renna y Esparza). Sección de conclusiones al final.
- El informe **no debe incluir definiciones teóricas ni el significado de los parámetros de los métodos vistos en clase**. Las explicaciones didácticas pedidas arriba se dan en la conversación; en el notebook solo van la justificación de decisiones y la interpretación de resultados.
- Pocos gráficos y representativos. Cada uno debe llevar una explicación de lo observado y si coincide con la hipótesis previa.
