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
  - `U3_Clustering_Wheat.ipynb`: `KMeans(n_clusters=k, random_state=42)`, `AgglomerativeClustering(n_clusters=k, linkage='ward')`, `scipy.cluster.hierarchy.linkage` + `dendrogram`, `silhouette_score` / `silhouette_samples` (k de 2 a `max_k`), DBSCAN y HDBSCAN. **GAP se programa a mano** con `calculate_intra_cluster_dispersion(X, k)`, que compara `log(inercia de referencia) - log(inercia real)` usando 10 conjuntos de referencia, con `max_k = 10` y `optimal_k = np.argmax(gaps) + 1`. Hay una versión para K-means y otra para jerárquico. Reutilizar ese enfoque **con las correcciones** de la sección de la consigna 5.
  - `Unidad3.py` (script del profesor, con `wheat.csv` y `Mall_Customers.csv`; `CC GENERAL.csv` no lo usa ningún archivo): gráfico 3D de clusters con matplotlib, paleta `husl` y centroides con `marker='*'`; codo con `.score()`; dendrograma + `AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')` + `silhouette_score`; GAP con `gap_statistic.OptimalK`. **Esa librería no instala en este entorno** (falla al construir en Python 3.13), por eso se usa la función manual. El script no corre tal cual: termina con una URL suelta y trabaja sin estandarizar.
- `tp1/tp1_mineria_renna_esparza.ipynb`: entrega. Usa el kernel del venv `.entorno`. El avance está en la sección **Estado del TP1**, más abajo.

## Particularidades del dataset (`penguins_size.csv`)

No es el CSV original de Kaggle; está modificado:
- Separador `;`: hay que leerlo con `pd.read_csv(..., sep=';')`.
- Columnas en español: `Especie`, `Longitud Culmen (mm)`, `Profundidad Culmen (mm)`, `Longitud Aleta (mm)`, `Masa corporal (g)`, `Sexo`. No existe la columna de isla.
- Tiene dos columnas finales sin nombre y totalmente vacías (vienen del `;;` al final de cada línea), que hay que eliminar.
- 347 filas. Las 4 numéricas tienen 2 faltantes cada una (filas completamente vacías salvo la especie). `Sexo` tiene 10 vacíos y 1 valor inválido `'.'`.
- Especies: `Adelie Penguin` (154), `Chinstrap penguin` (68) y `Gentoo penguin` (125). Ojo: las mayúsculas no son consistentes.

## Entorno

El venv es `.entorno`, en la raíz del repo (Python 3.13, Windows). Git lo ignora por el `.gitignore` que crea el propio venv adentro. Los comandos se corren desde la raíz del repo. Las dependencias del TP1 están fijadas con versión en `requirements.txt`:

```powershell
.entorno\Scripts\python.exe -m pip install -r requirements.txt
```

Si se agrega una librería, fijar su versión en `requirements.txt`. `umap-learn`, `hdbscan` y `kagglehub` se usan en los notebooks de teoría pero quedaron afuera porque el TP1 no los necesita.

Ejecutar el notebook completo desde la consola (`nbclient`). Con `--output` la copia ejecutada se guarda aparte y la entrega no se modifica; con `--inplace` se sobrescribe. El kernel arranca en `tp1/`, por eso el notebook lee `penguins_size.csv` con ruta relativa. **`PYTHONUTF8=1` es obligatorio**: sin esa variable `nbclient` lee el archivo como cp1252 y rompe todos los acentos (con `--inplace` arruina la entrega; se detecta buscando `Ã` en el `.ipynb`):

```powershell
$env:PYTHONUTF8 = "1"; .entorno\Scripts\jupyter-execute.exe tp1\tp1_mineria_renna_esparza.ipynb --output C:\ruta\temporal\salida.ipynb
```

Después de modificar código hay que re-ejecutar el notebook con `--inplace` para que las salidas guardadas coincidan con las celdas.

Leer una página de los PDF de teoría (`pypdf`):

```powershell
.entorno\Scripts\python.exe -c "from pypdf import PdfReader; print(PdfReader('tp1/teoria/Mineria_Datos_U2.pdf').pages[45].extract_text())"
```

Smart App Control de Windows llegó a bloquear los `.pyd` de scipy y scikit-learn (`ImportError: DLL load failed ... Una directiva de Control de aplicaciones bloqueó este archivo`). Los autores lo desactivaron el 29/09/2026 y volvió a funcionar. Si reaparece un error así, no es del código.

## Requisitos de formato de la entrega (del enunciado)

- Cabecera con año, materia e integrantes (Renna y Esparza). Sección de conclusiones al final.
- El informe **no debe incluir definiciones teóricas ni el significado de los parámetros de los métodos vistos en clase**. Las explicaciones didácticas pedidas arriba se dan en la conversación y en `RepasoParcial.md`, nunca en el notebook.
- Pocos gráficos y representativos. Cada uno debe llevar una explicación de lo observado y si coincide con la hipótesis previa.

### Redacción del notebook (regla de los autores)

- **Los comentarios son solo sobre lo observado y si es coherente con la hipótesis previa, nada más.** No explicar cómo funciona un método, qué mide una métrica ni por qué "era previsible" un resultado. Se admite una frase corta para justificar una decisión (por qué se elige un valor, por qué se elimina una fila).
- **Solo se comenta lo que se ve en el notebook.** Todo número citado tiene que poder leerse en una figura, una tabla o una salida impresa del propio notebook. No citar valores calculados aparte (rangos, medias por grupo, distancias mínimas, conteos de vecinos, etc.): los autores no pueden defender de dónde salen. Si un número hace falta, primero se muestra en una celda; si no, se describe lo que se ve ("queda pegado", "se superponen").
- **"Proyección" solo para PCA**, que es lineal. Los resultados de Isomap y t-SNE no son proyecciones: se los llama "representación en dos dimensiones" o "el gráfico de Isomap / t-SNE".
- No agregar apartados que el enunciado no pide (por ejemplo, se quitó el de dependencia de la semilla en t-SNE; en el TP2, la comparación de escalado en k-NN, la Tabla 7 de estandarización y las barras de exactitud).
- **Escribir como estudiantes:** primera persona ("pensábamos que", "en la Figura 3 vemos que"), frases cortas y pocas negritas. Sin muletillas ("es decir", "ahora bien", "justamente", "por completo") ni explicaciones de causas que el notebook no muestra.
- **Código al estilo de la cátedra:** `enumerate` con `axes.flatten()[i-1]`, `ConfusionMatrixDisplay.from_predictions` con `cmap=plt.cm.Blues` y funciones chicas como `tableResult`. Antes de usar un método o parámetro, buscarlo en `teoria/`; si no está y hace falta, justificarlo en una línea. En el TP2 quedan fuera de clase `KBinsDiscretizer`, `CategoricalNB` y `ccp_alpha` (los pide el enunciado), `OneHotEncoder` (filmina 28 de la U5), `pd.crosstab`, `get_depth()`, `get_n_leaves()` y `neg_mean_squared_error`.

### Numeración de figuras y tablas

- **Figuras:** numeración única y corrida en todo el notebook (`Figura 1`, `Figura 2`, ...). El número va en el título del gráfico: `plt.title("Figura 7. ...")`, o `plt.suptitle(...)` en las grillas y en el `pairplot` (ahí con `y=1.02`). Una grilla de paneles cuenta como una sola figura.
- **Tablas:** numeración única y corrida, independiente de la de figuras. Cuenta como tabla todo `DataFrame` que se muestra y toda tabla escrita en markdown; no cuentan las salidas de texto (`value_counts()`, `isna().sum()`, `info()`, prints).
  - `DataFrame`: `print("Tabla 3. ...")` como última línea antes de la expresión que lo muestra, para que el título quede justo arriba.
  - Markdown: una línea `**Tabla 10.** ...` antes de la tabla.
- En el texto se las cita por número ("en la Figura 2", "(Tabla 13)") en lugar de "el gráfico anterior".
- Al insertar una figura o tabla en el medio hay que renumerar las siguientes y sus citas. Hoy el notebook llega hasta la **Figura 20** y la **Tabla 26** (fin de la consigna 6).

## Forma de trabajo

Los autores piden avanzar **paso por paso** para entender cada decisión:
1. Ejecutar primero el código en el venv (`.entorno\Scripts\python.exe`) y revisar la salida real. No escribir interpretaciones con números que no se hayan visto.
2. Agregar al notebook las celdas del paso: código + markdown con la justificación y la interpretación.
3. Explicar en el chat qué hace cada línea y por qué se eligió cada parámetro.
4. Esperar la aprobación antes de pasar al paso siguiente.

Las decisiones que modifican el dataset las toman los autores: consultarlas antes de implementarlas.

**`RepasoParcial.md` (raíz del repo):** material de estudio para el parcial, con la explicación **para principiantes** de todos los conceptos aplicados en los TPs. Es la primera vez que los autores ven estos temas.
- **Cada vez que aparece un tema o concepto nuevo, hay que agregarlo al archivo con su explicación**, en el mismo paso en que se usa.
- Debe cubrir estrictamente todos los temas de los TPs, con ejemplos de los datos propios y los números reales obtenidos.
- Vive fuera de `tp1/` porque va a abarcar también los trabajos prácticos siguientes.
- La sección final lista los temas todavía no vistos; al explicarlos, se mueven al cuerpo del archivo.

Convenciones del notebook:
- Variables que se arrastran entre celdas: `df` (dataset limpio, 342 filas), `num_cols` (las 4 medidas numéricas), `X` (5 características), `y` (`Especie`) y `X_std` (`X` estandarizada). **Las consignas 2 a 6 trabajan sobre `X_std` y usan `y` solo para colorear y comparar.**
- Los ids de celda llevan el prefijo del paso: `cab-`, `cons-`, `conf-`, `imp-`, `carga-`, `estr-`, `desc-`, `cat-`, `nul-`, `dup-`, `dist-`, `corr-`, `out-`, `cod-`, `std-`, `cierre-` en la consigna 1, y un único prefijo por método desde la consigna 2 (`pca-`, `iso-`, `tsne-`, `km-`, `jer-`, y `conc-` para las conclusiones), numerado en orden (`pca-01`, `pca-02`, ...).
- Antes de cada gráfico va una celda markdown con la **hipótesis previa**, y después otra con la interpretación que la confirma o la refuta (lo pide el enunciado).
- Se usa pandas 3: las columnas de texto tienen dtype `str` y no `object` como en los notebooks de clase.
- **Nombres de las componentes:** `scikit-learn` numera las columnas de salida desde cero (`pca0`, `pca1`, ...). Apenas se ajusta un método de reducción, sus columnas se renombran con el prefijo del método y numeradas desde 1: `PC1`, `PC2`, ... en PCA; `ISO1`, `ISO2` en Isomap; `TSNE1`, `TSNE2` en t-SNE.

## Devolución del profesor sobre el TP1 (nota 7)

Recibida el 16/10/2026, con el TP1 ya entregado. **Decisión de los autores: el TP1 no se corrige.** Las observaciones se aplican de acá en adelante (TP2 y parcial); los conceptos involucrados están en la sección 25 de `RepasoParcial.md`.

Aprobado sin objeciones: el análisis descriptivo, el tratamiento de nulos y duplicados, Isomap, y la justificación de la cantidad de componentes de PCA.

| Observación | Qué dijo | Criterio a aplicar de ahora en más |
|---|---|---|
| **Codo en PCA** | El quiebre está en la tercera componente, así que había que tomar **3**, no 2 | En el método del codo **la componente del quiebre se incluye**. El notebook dice "el quiebre se produce en la tercera componente, por lo que conservamos las anteriores" (celda `pca-13`): eso es lo que marcó mal |
| **Codo en K-means** | El quiebre se produce en k=4 o 5, no en 6 como dice la celda `km-04`. Aclaró que está bien pasar a otros criterios si el codo no alcanza | Misma regla: el k del quiebre se incluye. El k=6 final, elegido por Silhouette y GAP, no fue objetado |
| **Barridos de t-SNE** | Pidió variar la perplejidad con las iteraciones fijas y variar las iteraciones con la perplejidad fija, para elegir **la mejor combinación** de los dos parámetros; dijo que no aclaramos los valores fijos | Es lo que hicimos, pero los valores fijos figuraban solo en el `suptitle` de las Figuras 11 y 12 y el markdown decía "con el resto de los parámetros fijos". Hay que **escribir el valor fijo en el texto** de cada barrido y cerrar explicitando la combinación elegida |
| **La medida KL** | No se vio en clase, y la usamos para elegir la cantidad de componentes sin explicar qué es ni qué significa que mejore | **No usar métricas que no se vieron en clase** para justificar decisiones. Explicarla tampoco era salida, porque el enunciado prohíbe incluir teoría: se decide con los criterios de clase y con lo que se observa en los gráficos |
| **Clustering jerárquico** | "No imprimen cómo quedan los grupos" | Mostrar la composición de los clusters **en todos los métodos**, con tabla y gráfico. Las Tablas 24 y 25 sí los cruzan con especie y sexo, pero faltó la figura equivalente a la 17 de K-means |

## TP2 (unidades 4 y 5)

Enunciado: `tp2/TP2.pdf`. Entrega: `tp2/tp2_mineria_renna_esparza.ipynb`, un solo archivo.

**Parte A, regresión** (`1000_Companies.csv`, estimar `Profit`): EDA y preparación con dos particiones (80/20 y 70/30); `DecisionTreeRegressor` explorando `max_depth`, `min_samples_leaf` y `min_samples_split`; gráfico del árbol; MAE, MSE y RMSE en ambos test.

**Parte B, clasificación** (`drugType.csv`, estimar `Droga`): lo mismo con particiones **estratificadas**; `DecisionTreeClassifier` con poda `ccp_alpha` (árbol antes y después); Naive Bayes (discretización por cuantiles con 3 y 5 intervalos contra `GaussianNB`); k-NN (5 valores de `n_neighbors`, métricas euclidean y manhattan); tabla comparativa de los tres y elección justificada. Métricas: Precision y Recall **macro**, Accuracy y matriz de confusión.

**`random_state=63` en todo** (constante `SEMILLA` en el notebook). Máximo 3-4 gráficos por actividad.

### Datos

| | `1000_Companies.csv` | `drugType.csv` |
|---|---|---|
| Tamaño | 1000 × 5 | 200 × 6 |
| Objetivo | `Profit` (continua) | `Droga` (5 clases) |
| Numéricas | RyD_Spend, Administration, MarketingSpend | Edad, Na_a_K |
| Categóricas | State (nominal) | Sexo (binaria), BP y Colesterol (ordinales) |
| Nulos | 0 | 0 |
| Duplicados | 1 | 0 |
| Particularidad | `Profit ≈ 0.871 × RyD_Spend + 48.403`: casi determinista. `RyD_Spend` y `MarketingSpend` correlacionan 0.98 entre sí | **Clases desbalanceadas**: DrogaY 91, DrogaX 54, DrogaA 23, DrogaC 16, DrogaB 16 |

### Lo que impone la teoría de las unidades 4 y 5

- **U4, filmina 70: estandarizar DESPUÉS de separar train/test.** El `fit` del escalador, del codificador, del imputador y del discretizador va **solo con el entrenamiento**; al test se le aplica `transform`. Es la diferencia central con el TP1.
- **Los notebooks de árboles (U4) no estandarizan**: ni importan `StandardScaler`. Los de U5 (k-NN, Bayes) sí, porque usan distancias. Por eso en el TP2 se estandariza en la preparación (lo pide el enunciado y lo necesita la Parte B), pero **los árboles se entrenan con los valores originales**, para que los umbrales del gráfico se lean en las unidades de los datos.
- **Patrón de preparación de la U5** (`U5_KNNClassifier_heart.ipynb`, celdas 14, 24 y 31): `train_test_split` → `enc.fit(Xtrn)` + `transform` en ambos → `scaler.fit(Xtrn[num_cols])` + `transform` en ambos. Solo se escalan las numéricas.
- **Validación cruzada:** `GridSearchCV(cv=5)` se usa en U4 clasificación (`scoring='f1_weighted'`), U5 k-NN (`scoring='accuracy'`) y U5 SVM (sin `scoring`). El **único** que no la usa es U4 regresión, que parte en tres (train/val/test) y recorre la grilla con `itertools.product`. Las filminas 40-47 de la U4 explican k-fold y Monte Carlo, y dicen "K típicamente 5 o 10".
- **Grilla de la U4 regresión** (celda 83), que es la que usamos: `max_depth: [3, 5, 10, None]`, `min_samples_split: [2, 5, 10]`, `min_samples_leaf: [1, 5, 10]`.
- **Graficar árboles:** la teoría usa `export_graphviz` + `graphviz`, que **no está instalado**. Se usa `plot_tree` de sklearn, que los propios notebooks importan.

### Convenciones propias del TP2

- Prefijos de celda: `pa-` (Parte A) y `pb-` (Parte B), numerados en orden. `cab-`, `conf-` e `imp-` como en el TP1.
- **Figuras y tablas se numeran desde 1**, independientes del TP1. El notebook llega a la **Figura 14** y la **Tabla 22**, más la celda `conc-01` de conclusiones. **El TP2 está completo.**
- El kernel arranca en `tp2/`, así que los CSV se leen con ruta relativa.
- Sin `n_jobs=-1` en `GridSearchCV`: en Windows deja tracebacks de joblib al cerrar los procesos y parecen errores en la entrega.

### Estado

| Actividad | Contenido | Estado |
|---|---|---|
| A1 | EDA, limpieza, codificación, particiones y estandarización | Hecho |
| A2 | Árbol de regresión: `GridSearchCV`, Figura 4 (árbol), Figura 5 (estimado contra real), Tabla 9 (MAE/MSE/RMSE) | Hecho |
| B3 | EDA de `drugType.csv`, desbalance (Figura 6, Tabla 12), numéricas y categóricas por droga (Figuras 7 y 8), atípicos, codificación ordinal y particiones estratificadas (Tabla 14) | Hecho |
| B4 | Árbol de clasificación: `GridSearchCV` con `f1_weighted` (Tabla 15), árbol sin podar (Figura 9), grilla de `ccp_alpha` (Tabla 16), árbol podado (Figura 10), métricas (Tabla 17) y matrices de confusión (Figura 11) | Hecho |
| B5 | Naive Bayes: discretización por cuantiles (Tabla 18), comparación de las configuraciones (Tabla 19) y matrices de confusión (Figura 12) | Hecho |
| B6 | k-NN: `GridSearchCV` sobre 5 valores de `n_neighbors` y 2 métricas (Tabla 20), curva de exactitud (Figura 13), métricas (Tabla 21) y matrices (Figura 14) | Hecho |
| B7 | Comparación de los tres (Tabla 22), elección justificada y sección de conclusiones (`conc-01`) | Hecho |

**Numeración:** las secciones del notebook siguen el número de actividad del enunciado (A1-A2 y B3-B7), no una cuenta propia de la Parte B.

**Resultados de la Parte B hasta ahora:**
- `Na_a_K` separa por completo a DrogaY (mínimo 15.0 contra máximo 14.6 del resto); `BP` define dónde aparece cada una de las otras; `Edad` separa DrogaA de DrogaB; `Sexo` no aporta. Los 8 atípicos de `Na_a_K` son todos DrogaY y se conservan.
- Codificación: `OrdinalEncoder` con orden explícito (`Sexo` F<M, `BP` Bajo<Normal<Alto, `Colesterol` Normal<Alto); `Droga` queda como texto. `preparar_pacientes(test_size)` devuelve sin escalar y escaladas (solo `Edad` y `Na_a_K`); el árbol usa las sin escalar.
- Árbol: 27 de 36 combinaciones empatan en f1 de validación cruzada (0.9868 en 80/20); el árbol crece solo hasta profundidad 4 con 6 hojas puras. Cortes: `Na_a_K <= 14.829` → DrogaY; `BP <= 1.5`; `Edad <= 50.5` (A/B); `BP <= 0.5` (normal → DrogaX); `Colesterol <= 0.5` (X/C).
- **Grilla del clasificador (decisión de los autores: se deja como está):** la misma de la regresión, porque el enunciado pide los mismos tres hiperparámetros; la del notebook de clase no los incluye. Se evaluó agregar `class_weight: [None, 'balanced']` (lo usa la U4): empata en f1 (0.9868 / 0.9729) y da 1.0 en test, pero arma árboles más grandes (9 y 8 hojas contra 6). También se vio que el f1 por profundidad es 0.628 / 0.767 / 0.860 / 0.987 / 0.987 (1 a 5): el árbol se satura en 4. Ninguno de los dos cambios se aplicó.
- Poda (solo en 80/20): grilla propia en `GridSearchCV`, sin `cost_complexity_pruning_path` porque no se vio en clase. `ccp_alpha` en 0 / 0.01 / 0.05 / 0.07 / 0.1 / 0.15 da f1 0.9868 / 0.9868 / 0.9868 / 0.8705 / 0.7665 / 0.6283. Se elige **0, no podar**: con 0.07 (Figura 10) se van `BP <= 0.5` y `Colesterol` y DrogaC deja de predecirse (hoja de 56 con 13 DrogaC).
- Test: exactitud, precisión macro y exhaustividad macro **1.0 en las dos particiones**; matrices diagonales.
- **Naive Bayes (actividad 5):** `CategoricalNB` con 5 intervalos es el mejor (exactitud 0.9250 y 0.9167), seguido de `GaussianNB` (0.8500 en ambas) y `CategoricalNB` con 3 intervalos (0.8250 y 0.8500). Ninguno llega al árbol. Observación: los cortes por cuantiles de `Na_a_K` dejan el valor 15 **dentro** de un intervalo (11.42-18.3 con 3; 12.81-16.12 con 5), así que mezclan DrogaY con el resto. Todos los errores son DrogaY clasificada como otra droga: 3 en 80/20 y 5 en 70/30. La clase **no discretiza**, así que `KBinsDiscretizer` se usa porque lo pide el enunciado; el patrón de `GaussianNB` sobre datos escalados sí sale de la U5.

**Resultados de k-NN y comparación final:**
- k-NN (actividad 6): mejores configuraciones distintas por partición (3 vecinos euclídea en 80/20, 9 Manhattan en 70/30), exactitud 0.9000 y 0.9167. La elección es inestable: en la 80/20 dos combinaciones empatan en 0.8875 y seis quedan en 0.8688. La comparación con y sin escalado (0.9000 contra 0.6500) **se quitó del notebook** a pedido de los autores; queda en `RepasoParcial.md`.
- Comparación (actividad 7): el árbol gana en las tres métricas y en las dos particiones (1.0 contra 0.8859-0.9667). Entre Bayes y k-NN no hay diferencia clara. Se elige el árbol también por interpretabilidad. El árbol no tiene errores; en Bayes y k-NN se concentran en DrogaY.

**Decisiones de los autores en la Parte A:**

| Problema | Decisión | Justificación |
|---|---|---|
| 1 fila duplicada | `drop_duplicates()` → 999 | Si quedara repartida entre train y test, el modelo sería evaluado con una fila que ya vio |
| 3 filas con `Profit` atípico (las últimas del archivo: 997, 998, 999) | **Eliminadas** con el criterio IQR sobre `Profit` → 996 filas | Rompen la relación que siguen las otras 996 y, por ser la variable objetivo, distorsionan la medición del error |
| 22 atípicos en `Administration` | **Se conservan** | Es un predictor, no el objetivo; los árboles parten por umbrales |
| `State` | One-hot (3 columnas) | No tiene orden: codificarlo 0/1/2 daría distancias falsas en k-NN |
| Búsqueda de hiperparámetros | `GridSearchCV(cv=5)` con la grilla de la U4 | Decisión de los autores; la herramienta se usa en 3 de los 4 notebooks de clase |

**Resultados de la Parte A:** mejores parámetros 80/20 `max_depth=10, min_samples_leaf=1, min_samples_split=10` (141 hojas) y 70/30 `max_depth=10, min_samples_leaf=5, min_samples_split=2` (106 hojas). MAE 997.20 / 949.77; RMSE 3585.19 / 2890.49. `RyD_Spend` concentra el 98.9% de la importancia y las tres columnas de `State` quedan en 0.

### Ejecutar el notebook del TP2

Igual que el del TP1, con `PYTHONUTF8=1` obligatorio. Tarda unos 15 segundos.

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

### Consigna 2: PCA (terminada)

| Paso | Contenido | Estado |
|---|---|---|
| 1 | Hipótesis previa (redundancia entre aleta y masa → pocas componentes) | Hecho |
| 2 | `from sklearn.decomposition import PCA` en imports; `pca = PCA().set_output(transform="pandas")`, `X_pca = pca.fit_transform(X_std)` y renombrado de columnas a `PC1`..`PC5`. Sin `Pipeline` porque `X_std` ya está estandarizada | Hecho |
| 3 | Criterio 1: varianza acumulada, `var_exp`/`var_cum` + gráfico de barras y línea (U2 celdas 93-94, sin cambios) → 2 componentes (84.6%) | Hecho |
| 4 | Criterio 2: Kaiser, tabla con `explained_variance_` y `explained_variance_ratio_` → 2 (2.87 y 1.38) | Hecho |
| 5 | Criterio 3: codo (U2 celda 101) → quiebre en PC3, se conservan 2 | Hecho |
| 6 | Elección: los 3 coinciden en 2; se argumenta que varianza acumulada representa mejor porque cuantifica lo conservado (84.6%) | Hecho |
| 7 | Tabla de cargas `pca.components_`: PC1 tamaño, PC2 sexo + profundidad, PC3 longitud del culmen | Hecho |
| 8 | Scatter 2D `PC1` vs `PC2` con `hue=y` (U2 celda 107): Gentoo separada, Adelie/Chinstrap superpuestas, cada especie partida en dos nubes por sexo | Hecho |
| 9 | 3D con `PC1`, `PC2`, `PC3` (U2 celda 109 corregida; colores con `sns.color_palette('Set1')` para que coincidan con el 2D): separa Adelie de Chinstrap | Hecho |
| 10 | Cierre de la consigna 2 | Hecho |

Resultados esperados (calculados sobre `X_std`): varianza explicada 57.2%, 27.5%, 9.6%, 3.7%, 2.1% (acumulada con 2 componentes: 84.6%); eigenvalues 2.87, 1.38, 0.48, 0.18, 0.11. Los 3 criterios dan 2 componentes. PC1 ≈ tamaño; PC2 ≈ profundidad del culmen + sexo; PC3 ≈ longitud del culmen, que es la variable que separa Adelie de Chinstrap.

**Errores en `U2_Reducción_de_la_dimensionalidad.ipynb` que no hay que copiar:**
- Celda 98: aplica Kaiser como "varianza explicada ≥ 0.1"; el criterio correcto es eigenvalue (`explained_variance_`) > 1, como dice la celda 91. Con estos datos ambos dan 2.
- Celda 109: el gráfico 3D usa `pca1`, `pca2`, `pca3` y se saltea `pca0`, porque sklearn numera las columnas desde 0.

### Consigna 3: Isomap (terminada)

| Paso | Contenido | Estado |
|---|---|---|
| 1 | Hipótesis previa (¿separa Adelie de Chinstrap en 2D, cosa que PCA no logró?) | Hecho |
| 2 | Imports: `Isomap`, `kneighbors_graph`, `connected_components` | Hecho |
| 3 | Variación de vecinos: grilla 2x2 con `n_neighbors` en {5, 11, 30, 50}, 2 componentes | Hecho |
| 4 | Diagnóstico del aviso de grafo desconectado y su causa | Hecho |
| 5 | Variación de componentes: `reconstruction_error()` para 1 a 5 con 15 vecinos | Hecho |
| 6 | Gráfico 2D final con la configuración elegida | Hecho |
| 7 | Comparación con PCA y cierre | Hecho |

Resultados: Isomap **sí separa las tres especies en 2D**, a diferencia de PCA. Error de reconstrucción con 15 vecinos: 4.15 (1 comp.), 1.18 (2), 0.62 (3), 0.56 (4), 0.54 (5). Configuración elegida: **`n_neighbors=15`, `n_components=2`** (decisión de los autores).

**Hallazgo importante:** el grafo de vecinos queda **desconectado en 4 bloques hasta 20 vecinos** (se conecta recién con 30). Los bloques son machos y hembras, cruzados con Gentoo contra el resto: `Sexo`, al ser binaria, deja una distancia fija de 2 unidades tras estandarizar y ningún punto tiene vecinos del otro sexo. Sin esa columna serían 2 bloques. Por eso aparece el `UserWarning` de scikit-learn al correr Isomap con pocos vecinos: **es esperado y está explicado en el notebook**, no es un error.

Ojo al interpretar: `reconstruction_error()` solo es comparable entre distintos `n_components` con el mismo `n_neighbors`.

### Consigna 4: t-SNE (terminada)

| Paso | Contenido | Estado |
|---|---|---|
| 1 | Hipótesis previa, encadenada con Isomap | Hecho |
| 2 | Import de `TSNE`; parámetros de clase (`init='random'`, `random_state=42`, `method='exact'`) | Hecho |
| 3 | Variación de perplejidad: grilla 2x2 con 5, 15, 30 y 50 | Hecho |
| 4 | Variación de iteraciones: grilla 2x2 con 300, 500, 1000 y 2000 (perplejidad 15) | Hecho |
| 5 | Variación de componentes: KL con 2 y 3, sin figura | Hecho |
| 6 | Gráfico 2D final y comparación con PCA e Isomap | Hecho |

El apartado de dependencia de la semilla (3 valores de `random_state`) **se quitó a pedido de los autores**: no volver a agregarlo.

Configuración elegida (decisión de los autores): **`perplexity=15`, `max_iter=1000`, `n_components=2`**. Con 15 se ven seis grupos compactos, dos por especie, con Gentoo aislada y un contacto puntual entre un grupo de Adelie y uno de Chinstrap. Con 30 y 50 Adelie y Chinstrap se juntan. En el notebook esto se describe solo visualmente: **no citar distancias mínimas ni conteos de vecinos**, que no se calculan ahí.

KL medidos: perplejidad 0.492 / 0.361 / 0.228 / 0.125 (5 / 15 / 30 / 50); iteraciones 0.549 / 0.383 / 0.361 / 0.353 (300 / 500 / 1000 / 2000); componentes 0.361 (2) y 0.265 (3).

**Cuidados:**
- En sklearn 1.9 el parámetro es **`max_iter`**, no `n_iter`. El mínimo (250) devuelve un KL centinela gigantesco porque la optimización termina durante la fase inicial.
- El KL **no** es comparable entre perplejidades distintas (la perplejidad define la referencia); sí entre iteraciones y entre componentes.
- t-SNE es el paso más lento del notebook, que completo tarda alrededor de 1 minuto.

### Consigna 5: K-means (terminada)

Celdas `km-01` a `km-25`. Imports agregados: `KMeans` y `silhouette_score`.

| Paso | Contenido | Estado |
|---|---|---|
| 1 | Hipótesis previa: óptimo mayor que 3, posiblemente 6 (por los seis grupos de las Figuras 10 y 13) | Hecho |
| 2 | Variación de k: inercia para k=1..10 y gráfico del codo (U3 celda 36) → Figura 14 | Hecho |
| 3 | Silhouette para k=2..10 con `calculate_silhouette` (U3 celdas 79-80) → Figura 15 | Hecho |
| 4 | GAP para k=1..10 con `calculate_intra_cluster_dispersion` (U3 celdas 75-77), corregida → Figura 16 | Hecho |
| 5 | `DataFrame` `resultados` con inercia, Silhouette y GAP por k → Tabla 19 | Hecho |
| 6 | `kmeans_3` y `kmeans_6` cruzados con especie y sexo (`pd.crosstab`) → Tablas 20 y 21 | Hecho |
| 7 | 3D con k=6 sobre longitud del culmen, masa corporal y profundidad del culmen, con centroides → Figura 17 | Hecho |
| 8 | Resumen → Tabla 22 | Hecho |

Resultados (con `random_state=42` y `np.random.seed(42)`):

| k | Inercia | Silhouette | GAP |
|---|---|---|---|
| 2 | 907.2 | 0.443 | 0.723 |
| 3 | 593.3 | 0.450 | 1.003 |
| 4 | 403.1 | 0.501 | 1.259 |
| 5 | 300.9 | 0.513 | 1.436 |
| **6** | **233.2** | **0.515** | **1.605** |
| 7 | 216.7 | 0.467 | 1.587 |
| 10 | 172.3 | 0.365 | 1.601 |

- **Óptimo k=6 por los dos índices, pero por poco margen**: Silhouette casi empata con k=4 y k=5, y el GAP queda plano de 6 en adelante (k=10 da 1.601). En el notebook se describe así, sin presentarlo como un máximo contundente.
- **k=3 no recupera las especies**: cluster 1 = Gentoo (123), cluster 0 = machos de Adelie y Chinstrap (73 + 34), cluster 2 = hembras (78 + 34).
- **k=6 = especie × sexo**: Gentoo machos 65 / hembras 58, Adelie machos 71 / hembras 77, Chinstrap machos 34 / hembras 32. Solo 5 de 342 fuera de su grupo (3 Adelie en clusters de Chinstrap, 2 Chinstrap hembras con las hembras de Adelie).
- Es otra consecuencia de haber incluido `Sexo` como característica, igual que el grafo desconectado de Isomap.

**Errores del código de clase que no hay que copiar** (decisiones aprobadas por los autores; aplican también a la consigna 6):
- U3 celda 75: la función de GAP hace `kmeans.fit(X_std)` e ignora su argumento `X`. Se corrige a `kmeans.fit(X)`.
- U3 celdas 76 y 84: la referencia sale de `np.random.rand(*X.shape)`, en [0, 1], mientras `X_std` va de −2 a 3. Así el GAP da negativo, creciente y óptimo k=10. Se genera dentro del rango de cada característica: `np.random.rand(*X_std.shape) * (maximos - minimos) + minimos` (filmina 41 de la U3: uniforme sobre el rectángulo que contiene a los datos).
- Ninguno fija semillas: se agregan `random_state=42` en cada `KMeans` y `np.random.seed(42)` antes del bucle del GAP.
- `Unidad3.py`: dibuja los centroides dentro del bucle de clusters. En el notebook se dibujan una sola vez, afuera.

Para el 3D se probaron varias combinaciones de atributos; la que mejor distingue los seis clusters con la vista por defecto es x = longitud del culmen, y = masa corporal, z = profundidad del culmen.

### Consigna 6: clustering jerárquico (terminada)

Celdas `jer-01` a `jer-22`. Imports agregados: `AgglomerativeClustering`, `dendrogram` y `linkage`.

| Paso | Contenido | Estado |
|---|---|---|
| 1 | Hipótesis previa: 6 clusters, como K-means | Hecho |
| 2 | `Z = linkage(X_std, "ward")` + dendrograma truncado (`lastp`, `p=20`) con `plt.axhline(y=9)` (U3 celdas 51 y 53) → Figura 18 | Hecho |
| 3 | Silhouette para k=2..10 con `calculate_silhouette(X_scaled, k, linkage='ward')` (U3 celdas 87-88; `max_k=10` en vez de 15 para igualar el rango del GAP) → Figura 19 | Hecho |
| 4 | GAP para k=1..10 con `calculate_intra_cluster_dispersion(X, k, linkage='ward')` (U3 celdas 84-85), corregida → Figura 20 | Hecho |
| 5 | `resultados_jer` con Silhouette y GAP por k → Tabla 23 | Hecho |
| 6 | `etiquetas_6` y `etiquetas_7` cruzadas con especie y sexo → Tablas 24 y 25 | Hecho |
| 7 | Elección del número de clusters y resumen → Tabla 26 | Hecho |

Resultados (`np.random.seed(42)`):

| k | Silhouette | GAP |
|---|---|---|
| 4 | 0.501 | 1.325 |
| 5 | 0.511 | 1.488 |
| **6** | **0.515** | 1.677 |
| **7** | 0.465 | **1.684** |
| 8 | 0.409 | 1.663 |

- **El óptimo por GAP es 7, pero se concluye 6** (decisión de los autores): es el máximo de Silhouette, es el corte del dendrograma (las fusiones saltan de ~6.3 a ~11.9), el GAP de 6 y 7 difiere en 0.007, y el séptimo cluster solo parte en dos a los machos de Gentoo (33 y 32).
- **k=6 = especie × sexo**: Gentoo machos 65 / hembras 58, Adelie machos 73 / hembras 78, Chinstrap machos 34 / hembras 29. 5 hembras de Chinstrap quedan con las hembras de Adelie. Mismo resultado que K-means (5 fuera de su grupo en ambos, aunque no son los mismos pingüinos).
- En el dendrograma truncado aparece una etiqueta `168` sin paréntesis: es una muestra suelta (su índice), no un grupo de 168.

**Error adicional del código de clase** (U3 celdas 83-84): la dispersión se calcula como `X_np[labels] - centroids[labels]`, que usa las etiquetas como números de fila. Se corrige a `X_np - centroids[labels]`. Sin la corrección el GAP da valores sin sentido. `AgglomerativeClustering` es determinista y no lleva `random_state`.

### Conclusiones (terminadas)

Celda `conc-01`, la última del notebook: sección `## Conclusiones` que pide el enunciado. Recorre datos, reducción de la dimensionalidad, clustering, las hipótesis y el efecto de `Sexo`, citando figuras y tablas. Si se cambia algún resultado de las consignas, hay que revisarla.

**El TP1 está completo.** Las celdas de las consignas 5 y 6 se generaron con scripts temporales que no quedaron en el repo; para modificarlas, editar el `.ipynb` y re-ejecutar.

### PDFs de teoría (`tp1/teoria/Mineria_Datos_U2.pdf` y `U3.pdf`)

Los autores agregaron las filminas de las unidades. **Hay que tenerlas en cuenta junto con los notebooks.** Se leen con `pypdf` (comando en la sección **Entorno**).

Lo revisado de la U2 (71 páginas) **confirma** lo hecho en las consignas 2 a 4, y aporta:
- PCA (pág. 25): estandarizar siempre antes; el signo de los autovalores puede invertirse según el software.
- Isomap (pág. 41): k chico desconecta el grafo, k grande crea atajos; (pág. 38) ante un grafo desconectado se puede trabajar cada componente por separado.
- t-SNE (pág. 53-57): perplejidad típica 5-50 y menor que la cantidad de puntos; con 100 los clusters se fusionan; iteraciones del orden de 1000; si se corta antes aparecen formas "pellizcadas"; conviene correrlo varias veces; **no se pueden leer ni los tamaños de los clusters ni las distancias entre ellos**; los ejes no tienen significado.
- Comparación PCA / Isomap / t-SNE / UMAP (pág. 69-70).

De la U3 (58 páginas) se revisó lo de K-means e índices:
- K-means (pág. 16-19): algoritmo y criterios de parada. Ejemplo en Python (pág. 24-27), que es `Unidad3.py`.
- GAP (pág. 41): la referencia es una distribución uniforme sobre el rectángulo que contiene a los datos. Silhouette (pág. 42-43): rango de −1 a 1 y fórmula (b − a) / max(a, b).
- Pág. 47: GAP con `gap_statistic.OptimalK` para K-means y para jerárquico.

- Clustering jerárquico (pág. 28-37): aglomerativo y divisivo, dendrograma, enlaces `single`, `complete`, `average` y `centroid`, y cómo leer la altura de las uniones. Ejemplo en Python (pág. 45-46): dendrograma con `ward` y `AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')`.
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
