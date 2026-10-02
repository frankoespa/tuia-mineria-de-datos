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
- No agregar apartados que el enunciado no pide (por ejemplo, se quitó el de dependencia de la semilla en t-SNE).

### Numeración de figuras y tablas

- **Figuras:** numeración única y corrida en todo el notebook (`Figura 1`, `Figura 2`, ...). El número va en el título del gráfico: `plt.title("Figura 7. ...")`, o `plt.suptitle(...)` en las grillas y en el `pairplot` (ahí con `y=1.02`). Una grilla de paneles cuenta como una sola figura.
- **Tablas:** numeración única y corrida, independiente de la de figuras. Cuenta como tabla todo `DataFrame` que se muestra y toda tabla escrita en markdown; no cuentan las salidas de texto (`value_counts()`, `isna().sum()`, `info()`, prints).
  - `DataFrame`: `print("Tabla 3. ...")` como última línea antes de la expresión que lo muestra, para que el título quede justo arriba.
  - Markdown: una línea `**Tabla 10.** ...` antes de la tabla.
- En el texto se las cita por número ("en la Figura 2", "(Tabla 13)") en lugar de "el gráfico anterior".
- Al insertar una figura o tabla en el medio hay que renumerar las siguientes y sus citas. Hoy el notebook llega hasta la **Figura 17** y la **Tabla 22** (fin de la consigna 5).

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
- Los ids de celda llevan el prefijo del paso: `cab-`, `cons-`, `conf-`, `imp-`, `carga-`, `estr-`, `desc-`, `cat-`, `nul-`, `dup-`, `dist-`, `corr-`, `out-`, `cod-`, `std-`, `cierre-` en la consigna 1, y un único prefijo por método desde la consigna 2 (`pca-`, `iso-`, `tsne-`), numerado en orden (`pca-01`, `pca-02`, ...).
- Antes de cada gráfico va una celda markdown con la **hipótesis previa**, y después otra con la interpretación que la confirma o la refuta (lo pide el enunciado).
- Se usa pandas 3: las columnas de texto tienen dtype `str` y no `object` como en los notebooks de clase.
- **Nombres de las componentes:** `scikit-learn` numera las columnas de salida desde cero (`pca0`, `pca1`, ...). Apenas se ajusta un método de reducción, sus columnas se renombran con el prefijo del método y numeradas desde 1: `PC1`, `PC2`, ... en PCA; `ISO1`, `ISO2` en Isomap; `TSNE1`, `TSNE2` en t-SNE.

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

Consigna 6: sin empezar. Usa Silhouette + GAP con `AgglomerativeClustering(n_clusters=k, linkage='ward')` (U3 celdas 83-88), con las mismas correcciones al GAP.

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

Falta revisar la parte de clustering jerárquico al empezar la consigna 6.
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
