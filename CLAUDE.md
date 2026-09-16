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
- `tp1/tp1_mineria_renna_esparza.ipynb`: entrega. Usa el kernel del venv `.entorno`. El avance está en la sección **Estado del TP1**, más abajo.

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

## Forma de trabajo

Los autores piden avanzar **paso por paso** para entender cada decisión:
1. Ejecutar primero el código en el venv (`.entorno\Scripts\python.exe`) y revisar la salida real. No escribir interpretaciones con números que no se hayan visto.
2. Agregar al notebook las celdas del paso: código + markdown con la justificación y la interpretación.
3. Explicar en el chat qué hace cada línea y por qué se eligió cada parámetro.
4. Esperar la aprobación antes de pasar al paso siguiente.

Las decisiones que modifican el dataset las toman los autores: consultarlas antes de implementarlas.

Convenciones del notebook:
- Variables que se arrastran entre celdas: `df` (dataset limpio, 342 filas), `num_cols` (las 4 medidas numéricas), `X` (5 características), `y` (`Especie`) y `X_std` (`X` estandarizada). **Las consignas 2 a 6 trabajan sobre `X_std` y usan `y` solo para colorear y comparar.**
- Los ids de celda llevan el prefijo del paso: `cab-`, `imp-`, `carga-`, `estr-`, `desc-`, `cat-`, `nul-`, `dup-`, `dist-`, `corr-`, `out-`, `cod-`, `std-`, `cierre-`.
- Antes de cada gráfico va una celda markdown con la **hipótesis previa**, y después otra con la interpretación que la confirma o la refuta (lo pide el enunciado).
- Se usa pandas 3: las columnas de texto tienen dtype `str` y no `object` como en los notebooks de clase.

## Estado del TP1

### Consigna 1: análisis exploratorio, limpieza y estandarización (terminada)

| Paso | Contenido | Estado |
|---|---|---|
| 0 | Cabecera (año, materia, integrantes) | Hecho |
| 1 | Imports | Hecho |
| 2 | Carga del CSV y eliminación de columnas vacías | Hecho |
| 3 | `describe()`, categóricas, normalización de especies, `'.'` a nulo | Hecho |
| 4 | Valores faltantes | Hecho |
| 5 | Duplicados | Hecho |
| 6 | Distribuciones: histogramas + boxplots por especie en grilla 2x2 (U1 celdas 26 y 54, U3 celda 18) | Hecho |
| 7 | Correlaciones: `plot_correlation_heatmap` (U2 celda 15) + `sns.pairplot(hue='Especie', palette='Set1', diag_kind='kde')` (U2 celda 32) | Hecho |
| 8 | Outliers: conteo IQR global y por especie (U2 celda 50), sin eliminarlos. Sin gráfico nuevo: se usan los boxplots del paso 6 | Hecho |
| 9 | `Sexo` a 0/1 con `.map()` (FEMALE=0, MALE=1); `X = df.drop(columns="Especie")` (342 × 5), `y = df["Especie"]` (U1 celda 38) | Hecho |
| 10 | `StandardScaler().set_output(transform="pandas")` + `fit_transform(X)` → `X_std` (U1 celda 57). Media 0; `describe()` muestra desvío 1.001 por usar n − 1 | Hecho |
| 11 | Markdown de cierre del punto 1 | Hecho |

Consignas 2 a 6: sin empezar.

### Decisiones de limpieza tomadas

| Problema | Decisión | Justificación |
|---|---|---|
| 2 columnas vacías (`Unnamed: 6`, `Unnamed: 7`) | `dropna(axis=1, how="all")` | No contienen datos; vienen del `;;` al final de cada línea. |
| Nombres de especie inconsistentes | Diccionario + `.map()` a `Adelie`, `Chinstrap`, `Gentoo`, verificando antes con `set(...) - set(map.keys())` | Patrón de U2 celdas 42-44. `.map()` convierte en nulo cualquier valor que no esté en el diccionario. |
| Valor `'.'` en `Sexo` | `.replace(".", np.nan)` | No es una categoría real (U2 celda 71). |
| Filas 3 y 342 sin ninguna medida | `dropna(subset=num_cols, how="all")` | Imputarlas sería inventar un pingüino. No se usa `SimpleImputer` porque no son valores sueltos. |
| 9 nulos restantes en `Sexo` | Moda **por especie** con `groupby("Especie")["Sexo"].transform(lambda x: x.fillna(x.mode()[0]))` | U2 celda 67. **Salvedad:** en Adelie hay empate 74/74 y `mode()` desempata por orden alfabético, así que sus 5 filas quedaron FEMALE de forma arbitraria. En Gentoo la moda es real (62 contra 58) y se imputaron 4 filas como MALE. Está declarado en el notebook. |
| 3 filas duplicadas (índices 126/127, 63/206, 254/256) | `drop_duplicates()` | Coinciden las 5 variables con decimales: son cargas repetidas. El método no se vio en clase (pandas sí) y queda justificado en el notebook. |
| Outliers | Se muestran y **no se tocan** | 0 outliers IQR sobre el total. Dentro de cada especie hay 6 (filas 19, 28, 130, 190, 191, 254): apenas superan los límites y son coherentes con el sexo (altos en machos, bajos en hembras). Eliminarlos quitaría variabilidad real (U2 celda 57). |
| `Sexo` categórica | Se codifica a 0/1 y queda como 5ª característica | Decisión de los autores. |
| Conjunto de prueba | **No** se hace `train_test_split` | U1 celda 28: ese paso corresponde al TP2. |

**Estado del dataset tras el paso 5:** 342 muestras (347 − 2 filas sin medidas − 3 duplicados), sin nulos. Adelie 151, Gentoo 123, Chinstrap 68.
