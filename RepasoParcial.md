# Repaso para el parcial — Minería de Datos

Explicaciones para principiantes de **todos** los conceptos que vamos aplicando en los trabajos prácticos, con los ejemplos de nuestros propios datos.

- **Secciones 1 a 17:** lo del TP1, con el conjunto de pingüinos (no supervisado).
- **Secciones 18 en adelante:** lo del TP2, con las empresas y los medicamentos (supervisado).

Cada vez que aparece un tema nuevo en un TP, se agrega acá con su explicación.

**Índice**

1. [De qué se trata todo esto](#1-de-qué-se-trata-todo-esto)
2. [Cómo se organizan los datos](#2-cómo-se-organizan-los-datos)
3. [Tipos de variables](#3-tipos-de-variables)
4. [Medidas resumen](#4-medidas-resumen)
5. [Los gráficos y cómo leerlos](#5-los-gráficos-y-cómo-leerlos)
6. [Valores faltantes](#6-valores-faltantes)
7. [Duplicados](#7-duplicados)
8. [Valores atípicos (outliers)](#8-valores-atípicos-outliers)
9. [Correlación](#9-correlación)
10. [Codificación de variables categóricas](#10-codificación-de-variables-categóricas)
11. [Escalas y estandarización](#11-escalas-y-estandarización)
12. [PCA](#12-pca-análisis-de-componentes-principales)
13. [Isomap](#13-isomap)
14. [t-SNE](#14-t-sne)
15. [Clustering y K-means](#15-clustering-y-k-means)
16. [Clustering jerárquico](#16-clustering-jerárquico)
17. [Cómo funciona scikit-learn](#17-cómo-funciona-scikit-learn)
18. [Aprendizaje supervisado](#18-aprendizaje-supervisado)
19. [Entrenamiento y test](#19-entrenamiento-y-test)
20. [Fuga de datos](#20-fuga-de-datos-data-leakage)
21. [Validación cruzada y GridSearchCV](#21-validación-cruzada-y-gridsearchcv)
22. [Árboles de decisión](#22-árboles-de-decisión)
23. [Métricas de regresión](#23-métricas-de-regresión)
24. [Preguntas típicas de parcial](#24-preguntas-típicas-de-parcial)
25. [Lo que nos corrigió el profesor en el TP1](#25-lo-que-nos-corrigió-el-profesor-en-el-tp1)
26. [Temas que todavía no vimos](#26-temas-que-todavía-no-vimos)
27. [Resumen de una carilla](#27-resumen-de-una-carilla)

---

## 1. De qué se trata todo esto

**Minería de datos** es buscar patrones útiles dentro de un montón de datos.

Hay dos grandes familias de métodos:

| | Aprendizaje **supervisado** | Aprendizaje **no supervisado** |
|---|---|---|
| ¿Conoce la respuesta? | Sí, se le muestran ejemplos con la respuesta correcta | No, solo ve los datos |
| Qué hace | Aprende a predecir la respuesta | Busca estructura escondida |
| Ejemplos | Clasificación, regresión | PCA, clustering |

**Todo el TP1 es no supervisado.** Aunque sabemos la especie de cada pingüino, **no se la damos** a los algoritmos: la guardamos aparte (en `y`) y la usamos solo para pintar los gráficos y para comprobar al final si lo que el algoritmo encontró solo coincide con las especies reales.

Si le diéramos la especie, estaríamos haciendo trampa: encontraría los grupos porque se los dijimos.

**Variable objetivo:** la que queremos explicar o predecir. En nuestro TP es `Especie`.

---

## 2. Cómo se organizan los datos

Los datos son una tabla:
- Cada **fila** es una **muestra** o individuo (un pingüino).
- Cada **columna** es una **variable**, **característica** o *feature* (una medida).

En nuestro TP1, después de la limpieza: **342 filas × 5 características**.

**Dimensiones.** Si cada pingüino tiene 5 medidas, cada pingüino es **un punto en un espacio de 5 dimensiones**. No podemos dibujarlo (vemos 2 en un papel, 3 con esfuerzo), y de ahí nacen los métodos de reducción de la dimensionalidad como PCA.

**Convención de nombres:**
- `X` = la matriz de características (lo que ve el algoritmo).
- `y` = la variable objetivo (lo que dejamos afuera).

---

## 3. Tipos de variables

**Numéricas (o continuas):** son números que se pueden medir y promediar. Ejemplo: masa corporal 3750 g.

**Categóricas:** son etiquetas. Ejemplo: `Especie` (Adelie, Chinstrap, Gentoo) o `Sexo` (MALE, FEMALE).
- **Binarias:** categóricas con solo dos valores posibles, como `Sexo`.

**Por qué importa la diferencia:** los algoritmos hacen cuentas (sumas, distancias, promedios). Con "MALE" no se puede hacer una resta. Por eso las categóricas hay que **codificarlas** (punto 10) o dejarlas afuera.

Una trampa: que algo sea un número no significa que sea numérico. Si las especies estuvieran codificadas como 1, 2 y 3, no tendría sentido decir que el promedio es 2, ni que Gentoo (3) es "el triple" de Adelie (1). Es una categórica disfrazada de número.

---

## 4. Medidas resumen

Salen con `df.describe()`.

| Medida | Qué es | En nuestros datos |
|---|---|---|
| **count** | Cuántos valores no nulos hay | 345 (no 347: no cuenta los faltantes) |
| **media** (mean) | El promedio | Masa: 4204 g |
| **mediana** (50%) | El valor del medio: la mitad está por debajo y la mitad por arriba | Masa: 4050 g |
| **desvío estándar** (std) | Cuánto se alejan los datos del promedio, en promedio. Chico = apretados; grande = desparramados | Masa: 806 g |
| **mínimo y máximo** | El más chico y el más grande | Masa: 2700 y 6300 g |
| **cuartiles** (25%, 50%, 75%) | Cortan los datos ordenados en 4 partes iguales | Masa: 3550, 4050, 4750 |

**Percentil:** el valor que deja por debajo a ese porcentaje de los datos. El percentil 50 **es** la mediana. Los cuartiles son los percentiles 25, 50 y 75.

**Truco: media contra mediana.** Si son parecidas, los datos son más o menos simétricos. Si la media es **mucho mayor** que la mediana, hay valores grandes que la estiran (asimetría hacia la derecha).
- En las estrellas de la clase U2: media 10.497 K contra mediana 5.776 K → muy asimétrica.
- En nuestros pingüinos: 4204 contra 4050 → bastante simétrica.

**Por qué la mediana aguanta mejor:** si a 10 personas de sueldo normal les agregamos un millonario, la **media** se dispara, pero la **mediana** casi no se mueve. Por eso para imputar faltantes suele preferirse la mediana.

---

## 5. Los gráficos y cómo leerlos

### Histograma
Divide el rango en tramos y cuenta cuántos datos caen en cada uno. Muestra **la forma** de la distribución.

- **Un solo pico (unimodal):** la típica campana.
- **Dos picos (bimodal):** muy importante, porque casi siempre significa que **hay dos poblaciones mezcladas**.

En nuestros datos, la longitud de la aleta tiene un pico en 190 mm y otro entre 210 y 230: son los Gentoo separados del resto.

### Boxplot (diagrama de caja)
Resume la distribución en una caja:
- La **caja** va del cuartil 1 al cuartil 3: adentro está el **50% del medio** de los datos. Su ancho es el IQR.
- La **línea del medio** es la mediana.
- Los **bigotes** llegan hasta el dato más lejano que todavía no es atípico (a lo sumo 1.5 × IQR desde la caja).
- Los **puntitos sueltos** son los valores atípicos.

Sirve sobre todo para **comparar grupos**: un boxplot por especie muestra de una si una medida las distingue.

### Scatter (dispersión)
Un punto por muestra, con una variable en cada eje. Muestra la relación entre dos variables. Si además se **colorea** por categoría, se ve si los grupos se separan.

### Pairplot
Una grilla con el scatter de **todos los pares** de variables, y en la diagonal la distribución de cada una. Sirve para encontrar de un vistazo qué par separa mejor los grupos.

En nuestros datos, el par "longitud contra profundidad del culmen" separa las tres especies casi sin superposición.

### Heatmap de correlación
La matriz de correlación pintada con colores: azul fuerte = correlación positiva alta, rojo = negativa, blanco = cerca de cero. La diagonal siempre vale 1 (cada variable consigo misma).

---

## 6. Valores faltantes

Un **valor faltante** o **nulo** (`NaN`) es una celda vacía. Molestan porque la mayoría de los algoritmos no saben qué hacer con ellos.

**Tres caminos posibles:**

| Opción | Cuándo conviene | Riesgo |
|---|---|---|
| Eliminar la fila | Cuando la fila casi no tiene datos | Perder muestras |
| Eliminar la columna | Cuando a la variable le falta casi todo | Perder una variable entera |
| **Imputar** (completar) | Cuando falta poco | Inventar datos que sesgan |

**Imputar** es rellenar el hueco con un valor estimado:
- Numéricas: la **media** o la **mediana** de la columna.
- Categóricas: la **moda**, es decir, la categoría más frecuente.

### Imputar por grupo, no por columna
Es mejor completar usando el promedio **del grupo al que pertenece la muestra**, no el de toda la columna.

Ejemplo: si falta la masa de un pingüino y sabemos que es Gentoo, conviene usar la mediana de los Gentoo (5076 g), no la general (4204 g), porque los Gentoo son mucho más pesados. Completar con el valor general sesga los datos hacia el medio y desdibuja las diferencias entre grupos.

En el TP completamos `Sexo` con la moda **de cada especie**.

### Dos errores clásicos
1. **Rellenar categóricas con "Desconocido".** Estaríamos inventando una categoría que no existe en la realidad, y después el modelo intentaría predecirla. Un faltante no es una categoría.
2. **Imputar una fila que no tiene ningún dato.** Si a un pingüino le faltan las 5 medidas, completarlas todas con promedios es literalmente **inventar un pingüino**, que además cae justo en el centro de los datos y ensucia el análisis. Esas filas se eliminan.

### Faltantes disfrazados
No siempre aparecen como celda vacía: pueden venir como `'.'`, `'-'`, `'?'`, `-999` o `'N/A'`. Hay que detectarlos (mirando los valores únicos de cada columna) y convertirlos a nulo. En nuestro CSV, `Sexo` tenía un `'.'`.

---

## 7. Duplicados

Filas **exactamente iguales** en todas sus columnas, generalmente por un error de carga.

**Por qué molestan:** una fila repetida cuenta dos veces. Corre los promedios, infla la correlación de esa zona y, en clustering, actúa como si esa región tuviera el doble de densidad, atrayendo al centro del grupo.

**Cuidado:** que dos filas se parezcan no las hace duplicadas. Deben coincidir en todo. Si coinciden 4 medidas con decimales y además el sexo, es carga repetida y no dos animales distintos.

En el TP había 3 pares repetidos y los eliminamos.

---

## 8. Valores atípicos (outliers)

Un **outlier** es un valor muy alejado del resto.

### El criterio del IQR (el que usamos)
1. Se calculan el cuartil 1 (Q1) y el cuartil 3 (Q3).
2. **IQR = Q3 − Q1** (el ancho de la caja del boxplot).
3. Límites: **Q1 − 1.5 × IQR** y **Q3 + 1.5 × IQR**.
4. Todo lo que quede afuera es atípico.

Ejemplo con la masa de los Chinstrap: Q1 = 3487, Q3 = 3950, IQR = 463. Límites: 2794 y 4644. Un Chinstrap de 4800 g queda afuera: es atípico **para su especie**.

### Qué hacer con ellos
| Opción | Qué hace |
|---|---|
| Eliminarlos | Borra esas filas |
| **Winsorizing** | No las borra: **recorta** el valor hasta el límite (4800 pasaría a 4644) |
| Conservarlos | No toca nada |

### La decisión importante
**Un outlier no es automáticamente un error.** Antes de tocarlo hay que preguntarse si es un dato imposible (un pingüino de 50 kg: error de carga) o un valor real y extremo (un macho grande).

**La regla de oro (lección de la clase U2):** nunca eliminar valores que son **propios de una clase**. En el ejemplo de las estrellas, borrar los atípicos eliminaba casi toda una categoría de estrellas. Se borraba justo lo que hacía distinta a esa clase.

En nuestro TP **no eliminamos ninguno**: sobre el conjunto completo no hay atípicos, y los 6 que aparecen dentro de cada especie apenas pasan el límite y son coherentes con el sexo (los grandes son machos, los chicos hembras).

---

## 9. Correlación

El **coeficiente de correlación de Pearson** mide si dos variables se mueven juntas. Va de **−1 a 1**:

| Valor | Significa |
|---|---|
| **+1** | Cuando una sube, la otra sube siempre (relación perfecta) |
| **+0.87** | Cuando una sube, la otra casi siempre sube (aleta y masa, en nuestros datos) |
| **0** | No hay relación lineal |
| **−0.58** | Cuando una sube, la otra suele bajar |
| **−1** | Relación inversa perfecta |

### Para qué sirve acá
Si dos variables están muy correlacionadas, **dicen casi lo mismo**: hay información **redundante**. Eso es exactamente lo que PCA aprovecha para comprimir.

### Cuatro advertencias
1. **Solo mide relaciones lineales.** Dos variables pueden tener una relación fuerte en forma de U y dar correlación 0.
2. **No dice nada de la pendiente.** Una correlación de 0.9 no indica *cuánto* sube una cuando sube la otra.
3. **Correlación no es causalidad.** Que dos cosas se muevan juntas no significa que una cause la otra.
4. **Mezclar grupos puede invertir el resultado.** Ver abajo.

### La paradoja de Simpson (nos pasó en el TP)
La profundidad del culmen da **−0.47** con la masa sobre todo el conjunto: parecería que los pingüinos más pesados tienen el culmen menos profundo.

Pero calculando **dentro de cada especie**, da positiva en las tres: +0.58 en Adelie, +0.60 en Chinstrap y +0.72 en Gentoo.

¿Por qué? Porque los Gentoo son a la vez los más pesados **y** los de culmen más chato. Al mezclar las especies, esa diferencia entre grupos domina y **da vuelta** la tendencia.

**Moraleja:** una correlación calculada sobre grupos mezclados puede decir exactamente lo contrario de la verdad. Siempre conviene mirar también dentro de cada grupo.

---

## 10. Codificación de variables categóricas

Convertir etiquetas en números para que el algoritmo pueda operar.

**Variable binaria (dos categorías):** alcanza una sola columna con 0 y 1. Nosotros usamos FEMALE = 0 y MALE = 1. Qué número le toca a cada una es arbitrario y no cambia los resultados.

**Más de dos categorías:** hay que distinguir si la variable tiene orden o no.

| | Ejemplo | Cómo se codifica |
|---|---|---|
| **Ordinal** (tiene orden) | `BP`: Bajo < Normal < Alto | `OrdinalEncoder` con el orden explícito: 0, 1, 2 |
| **Nominal** (no tiene orden) | `State`: California, Florida, New York | **One-hot**: una columna 0/1 por categoría |

**Por qué one-hot para las nominales.** Codificar los estados como 0, 1 y 2 le estaría diciendo al algoritmo que Florida (1) está "en el medio" entre California (0) y New York (2), y que California está el doble de lejos de New York que de Florida. Eso es inventado. Con one-hot, `State` se reemplaza por tres columnas (`State_California`, `State_Florida`, `State_New York`) donde cada empresa tiene un 1 en la suya y 0 en las otras: todas quedan a la misma distancia entre sí.

A los **árboles** esto casi no les afecta, porque parten por umbrales. A **k-NN** sí, porque mide distancias: con la codificación 0/1/2 las distancias entre estados serían falsas.

⚠️ El codificador también se ajusta **solo con el entrenamiento** (ver la sección 20).

**Cuidado con `.map()`:** convierte en nulo, sin avisar, cualquier valor que no esté en el diccionario. Por eso antes de aplicarlo conviene verificar que no quede nada afuera:

```python
set(df["columna"]) - set(mi_diccionario.keys())   # tiene que dar set()
```

---

## 11. Escalas y estandarización

### El problema
Nuestras variables viven en escalas muy distintas:

| Variable | Rango |
|---|---|
| Profundidad del culmen | 13 a 21 mm |
| Masa corporal | 2700 a 6300 g |

Muchos algoritmos (PCA, K-means, Isomap, t-SNE) se basan en **distancias** o en **varianzas**. Una diferencia de 500 g pesa numéricamente muchísimo más que una de 3 mm, aunque en la biología del animal esos 3 mm puedan ser más importantes. Sin corregirlo, **la masa decide todo y el culmen no pesa nada**.

### La solución: estandarizar (z-score)
A cada valor se le resta la media de su columna y se lo divide por el desvío:

```
z = (valor − media) / desvío
```

Ejemplo: un pingüino de 3750 g, con media 4202 y desvío 801 → z = **−0.56**.

(Esa media es la del conjunto ya limpio, de 342 filas. En el punto 4 figura 4204 porque ese `describe()` se hizo antes de eliminar las filas sin medidas y los duplicados.)

Después de estandarizar, **todas las columnas tienen media 0 y desvío 1**, y se leen en la misma unidad: "cuántos desvíos me aparto del promedio". Un valor negativo está por debajo del promedio y uno positivo, por encima.

Ojo: estandarizar **no cambia la forma** de la distribución ni el orden de los datos. Solo los corre y los estira.

### Normalizar vs estandarizar
Suelen usarse como sinónimos, pero no son lo mismo:
- **Estandarizar (z-score):** media 0 y desvío 1. Es lo que hicimos.
- **Normalizar (min-max):** llevar todo al rango 0 a 1.

### El detalle del desvío 1.001
`StandardScaler` divide por **n** (desvío poblacional), mientras que `describe()` de pandas divide por **n − 1** (desvío muestral). Por eso el `describe()` de los datos estandarizados muestra 1.001 y no 1 exacto. Con 342 muestras la diferencia es despreciable y **no es un error**.

---

## 12. PCA (Análisis de Componentes Principales)

### El problema que resuelve
Tenemos 5 medidas por pingüino: cada uno es un punto en 5 dimensiones, y no hay forma de dibujarlo.

> ¿Puedo resumir esos 5 números en 2, perdiendo lo menos posible?

### La analogía de la sombra
Una fuente con forma de pescado, iluminada para proyectar su sombra:
- De frente a la punta, la sombra es un círculo chiquito: no se entiende qué es.
- De costado, la sombra muestra todo el largo y se reconoce la fuente.

Las dos sombras son planas, pero una conserva muchísima más información. **PCA busca automáticamente el mejor ángulo**, y "mejor" significa aquel donde los puntos quedan **más desparramados**.

### Varianza = información
Si todos los pingüinos pesaran igual, la masa no serviría para distinguir a ninguno. Una variable es útil cuando **varía**. Por eso PCA busca las direcciones de **mayor varianza**.

### Qué es una componente principal
Un **eje nuevo**, armado como una **receta** que mezcla las variables originales, cada una con un peso. Esos pesos se llaman **cargas** (*loadings*), y se leen en `pca.components_`.

Las recetas que encontró PCA con nuestros datos:

| | Long. culmen | Prof. culmen | Aleta | Masa | Sexo | Varianza explicada |
|---|---|---|---|---|---|---|
| **PC1** | 0.46 | −0.33 | 0.56 | 0.55 | 0.24 | **57.2%** |
| **PC2** | 0.17 | 0.66 | −0.10 | 0.04 | 0.73 | **27.5%** |
| **PC3** | 0.85 | 0.17 | −0.10 | −0.35 | −0.35 | 9.6% |
| PC4 | −0.15 | 0.65 | 0.37 | 0.37 | −0.53 | 3.7% |
| PC5 | −0.14 | 0.04 | 0.72 | −0.66 | 0.14 | 2.1% |

Cómo se interpretan (se miran las cargas más grandes en valor absoluto, con su signo):
- **PC1 = "tamaño general".** Suma aleta, masa y largo del culmen, y resta profundidad. PC1 alto = pingüino grande de culmen chato = **Gentoo** (promedio +1.93, contra −1.43 de Adelie).
- **PC2 = "culmen profundo y macho".** Casi toda su receta viene de la profundidad (0.66) y del sexo (0.73).
- **PC3 = "largo del culmen"** (0.85), que es justo la variable que distingue Adelie de Chinstrap.

### Dos reglas de las componentes
1. **Están ordenadas:** PC1 se queda con toda la varianza que puede, PC2 con la mayor parte de lo que sobró, y así.
2. **Son independientes entre sí** (no correlacionadas, u *ortogonales*): cada una aporta información **nueva**, no repite lo que ya dijo la anterior.

Las 5 componentes juntas suman el 100%: con todas, no se pierde nada, solo cambia el punto de vista. **La reducción ocurre al descartar las últimas.**

### Ejemplo con un pingüino
El pingüino de la fila 0 es un Adelie macho:
- En `X_std`: culmen −0.88, profundidad 0.79, aleta −1.42, masa −0.56, sexo 0.99.
- En PCA: **PC1 = −1.54, PC2 = 1.21**.

Traducción: es chico (PC1 negativo), de culmen profundo y macho (PC2 positivo). Pasamos de 5 números a 2 sin dejar de entender cómo es el animal.

### Siempre estandarizar antes
Si no, la variable con los números más grandes (la masa) se lleva toda la varianza y PC1 termina siendo "la masa" y nada más.

### Los 3 criterios para elegir cuántas componentes

**No hace falta que los tres coincidan: se puede elegir uno solo y justificarlo.**

| Criterio | Regla | Con nuestros datos | Resultado |
|---|---|---|---|
| **Varianza acumulada** | Quedarse con las necesarias para llegar al 75-80% | PC1 + PC2 = 84.6% | **2** |
| **Kaiser** | Conservar las de **eigenvalue > 1** | 2.87, **1.38**, 0.48, 0.18, 0.11 | **2** |
| **Codo (scree)** | Quedarse hasta la componente del quiebre, **incluida** | Quiebre en PC3 | **3** |

> ⚠️ En el TP1 entregado pusimos **2** en esa última casilla, razonando que si el quiebre está en PC3 se conservan "las anteriores". El profesor lo marcó como error: **la componente del quiebre se incluye**, así que ese criterio daba 3 (ver sección 25). La elección final de 2 componentes siguió siendo válida, porque alcanza con justificar uno de los tres criterios y los otros dos daban 2.

**Qué es un eigenvalue (valor propio):** la varianza de esa componente medida en "unidades de variable original". Como estandarizamos, cada variable original aporta exactamente 1. Por eso Kaiser dice: *si una componente vale menos de 1, aporta menos que una sola variable original y no justifica conservarla*.

Diferencia entre las dos columnas que devuelve sklearn:
- `explained_variance_` = **eigenvalues** (2.87, 1.38, …) → es el que usa Kaiser.
- `explained_variance_ratio_` = la misma información como **porcentaje** (57.2%, 27.5%, …) → es el de la varianza acumulada.

> ⚠️ El notebook de clase U2 aplica Kaiser (celda 98) como "varianza explicada ≥ 0.1", que **no** es el criterio de Kaiser enunciado en la celda 91. Con nuestros datos los dos dan 2 componentes, pero en el parcial conviene usar **eigenvalue > 1**.

### De eigenvalue a porcentaje: la cuenta

Los dos números que devuelve sklearn están ligados:

| | Qué es | Nuestros valores |
|---|---|---|
| `explained_variance_` | El **eigenvalue**: la varianza de esa componente | 2.87, 1.38, 0.48, 0.18, 0.11 |
| `explained_variance_ratio_` | Esa misma varianza como **proporción del total** | 0.572, 0.275, 0.096, 0.037, 0.021 |

La cuenta es una división. Los 5 eigenvalues suman 5.02 (con datos estandarizados, la suma da aproximadamente la cantidad de variables, porque cada una aporta varianza 1):

```
PC1: 2.87 / 5.02 = 0.572  →  57.2%
PC2: 1.38 / 5.02 = 0.275  →  27.5%
```

Por eso Kaiser compara contra **1**: una componente con eigenvalue menor que 1 aporta menos que una variable original suelta.

### ⚠️ El criterio del codo: dónde se corta exactamente

**Este punto nos lo corrigió el profesor en el TP1.** Nosotros escribimos que el quiebre estaba en la tercera componente y conservamos **2**; la corrección fue que había que tomar **3**.

**La regla: la componente donde se produce el quiebre se incluye.**

```
varianza
  │
0.57 ●
  │   ＼
0.27 ●  ＼          ← la caída fuerte termina ACÁ
  │     ＼
0.10 ●────●────●    ← a partir de acá la curva es plana
     1  2  3  4  5
           ↑
       el codo: se conservan 1, 2 y 3
```

El codo es el punto donde la curva **deja de caer y se aplana**. Ese punto marca la última componente que todavía aporta algo, así que entra.

Lo mismo vale para el codo en K-means (sección 15): el k del quiebre se incluye.

### Errores típicos con PCA

| Error | Por qué está mal |
|---|---|
| No estandarizar antes | La variable con los números más grandes se lleva toda la varianza |
| Interpretar PC1 como "una variable" | Es una **mezcla** de todas, con pesos: hay que leer las cargas |
| Creer que más varianza explicada = mejor separación de clases | PCA maximiza varianza, **no** separación. Nos pasó: PC3, con 9.6%, era la que distinguía Adelie de Chinstrap |
| Comparar eigenvalues entre datasets | Dependen de la cantidad de variables y de la escala |
| Olvidar que el signo es arbitrario | Un mismo análisis puede dar el gráfico espejado; no cambia nada |

### Los gráficos de PCA

**a) Varianza explicada y acumulada.** Las barras muestran cuánto aporta cada componente; la línea roja, cuánto se lleva acumulado. Se busca el primer punto de la línea que cruza el 75-80%.

**b) Gráfico del codo (scree).** La varianza de cada componente dibujada como curva que baja. Se busca **el quiebre**: el punto donde deja de caer bruscamente y se aplana. Se conservan las componentes **hasta la del quiebre, incluyéndola**. Se llama así porque la curva parece un brazo doblado.

En nuestra curva (0.57, 0.27, 0.10, 0.04, 0.02) el quiebre está en la tercera, porque de ahí en adelante los valores quedan casi alineados: el criterio da **3 componentes**. La tercera es, justamente, la que separa Adelie de Chinstrap.

**c) Scatter de PC1 contra PC2 coloreado por especie.** Cada punto es un pingüino en sus coordenadas nuevas. **PCA no conoce las especies**: los colores se agregan después. Si los colores quedan agrupados, significa que PCA encontró sola una estructura que coincide con las especies.

Cómo se lee un scatter de PCA, con lo que nos pasó a nosotros:
- **Grupos separados en el eje horizontal** = la primera componente los distingue. Gentoo quedó apartada a la derecha porque PC1 es el tamaño y es la especie más grande.
- **Colores mezclados** = esas clases no se distinguen con esas componentes. Adelie y Chinstrap quedaron encimadas, porque lo que las diferencia (el largo del culmen) vive en PC3.
- **Subgrupos dentro de un mismo color** = hay otra variable partiendo el grupo. Cada especie apareció dividida en dos nubes, arriba y abajo: son machos y hembras, porque PC2 está dominada por el sexo.

Esa última observación deja una enseñanza: **las variables que se incluyen en `X` deciden qué va a capturar cada componente**. Al incorporar `Sexo`, la segunda componente (27.5% de la varianza) se gastó en gran parte separando sexos en lugar de especies.

### Para qué sirve, y qué NO hace
**Sirve para:** visualizar datos de muchas dimensiones, eliminar redundancia, reducir ruido y preparar los datos para el clustering.

**No sirve para separar clases.** PCA maximiza **varianza**, no separación entre grupos. En nuestros datos se ve clarísimo:
- **PC2 está dominada por el sexo**, así que buena parte de la segunda dimensión se gasta en distinguir machos de hembras, no especies.
- **PC3 es el largo del culmen**, la única variable que separa Adelie de Chinstrap, y los tres criterios la descartan por explicar solo el 9.6%.

Conclusión para el parcial: **la componente más importante en varianza no es necesariamente la más útil para distinguir los grupos que nos interesan.**

---

## 13. Isomap

### El problema que PCA no puede resolver

PCA es **lineal**: sus componentes son rectas, así que solo puede mirar los datos desde otro ángulo, nunca "desenrollarlos".

Imaginen una serpentina enrollada en espiral. Dos puntos de la espiral pueden estar **muy cerca en línea recta**, porque una vuelta quedó encima de la otra, pero **lejísimos recorriendo la serpentina**. PCA mide en línea recta y los considera parecidos. Isomap, en cambio, mide **caminando por los datos**.

**Manifold learning** es la familia de métodos que asume que los datos, aunque estén en muchas dimensiones, en realidad viven sobre una superficie de menos dimensiones que está curvada (la serpentina enrollada). El objetivo es **desenrollar** esa superficie.

### Las dos distancias

| | Qué mide | Ejemplo |
|---|---|---|
| **Euclídea** | La línea recta entre dos puntos, atravesando el vacío | El túnel que atravesaría la montaña |
| **Geodésica** | El camino más corto **sin salirse de los datos**, saltando de vecino en vecino | El sendero que rodea la montaña |

Isomap usa la **geodésica**, y por eso capta estructuras curvas que PCA no ve.

### Cómo funciona, en 3 pasos

1. **Arma el grafo de vecinos.** A cada punto lo conecta con sus `n_neighbors` más cercanos.
2. **Calcula las distancias geodésicas**: el camino más corto entre cada par de puntos, pero solo pudiendo viajar por las conexiones del grafo.
3. **Ubica los puntos en 2 dimensiones** tratando de respetar lo mejor posible esas distancias.

### La distancia geodésica, con un ejemplo

Supongamos cuatro puntos y un grafo donde cada uno se conectó con sus vecinos más cercanos:

```
   A ──2── B ──3── C ──2── D
   └──────────12──────────┘
```

- **Distancia euclídea** entre A y D: **12**, la línea recta.
- **Distancia geodésica** entre A y D: se viaja por el grafo, A → B → C → D = 2 + 3 + 2 = **7**.

Isomap usa el 7. Si los puntos estuvieran sobre una superficie curva, ese camino refleja mucho mejor cuán "lejos" están realmente dentro de la estructura de los datos.

El algoritmo que busca el camino más corto entre todos los pares es Dijkstra o Floyd-Warshall (filmina 37 de la unidad 2).

### Los dos parámetros

**`n_components`:** cuántas dimensiones tiene la salida. Para graficar, 2.

**`n_neighbors`:** cuántos vecinos se conectan. Es **el** parámetro que hay que entender:

| | Qué pasa | Riesgo |
|---|---|---|
| **Muy chico** (3, 5) | Solo conecta lo más cercano: conserva mucho detalle **local** | El grafo puede quedar **partido en pedazos** entre los que no hay camino, y la representación se deforma |
| **Intermedio** | Equilibrio entre estructura local y global | Es lo que se busca |
| **Muy grande** (50) | Conecta puntos que no son realmente vecinos, con atajos que "atraviesan la montaña" | Las distancias geodésicas se parecen a las rectas y **el resultado se vuelve parecido a PCA**: se pierde lo no lineal |

### El grafo desconectado

Si `n_neighbors` es muy chico, pueden quedar grupos de puntos **sin ningún camino** que los una. La distancia geodésica entre ellos sería infinita, y sklearn avisa:

> `The number of connected components of the neighbors graph is 4 > 1`

No es un error: sklearn completa el grafo por su cuenta para poder seguir. Pero las distancias entre esos bloques quedan **inventadas**, así que la representación resultante no es confiable. **Ante ese aviso, hay que subir `n_neighbors`.**

### Los avisos que aparecen al correr Isomap (matrices dispersas)

Cuando el grafo queda desconectado, al ejecutar Isomap salen dos avisos distintos. **Los dos tienen la misma causa.**

**1. `UserWarning: The number of connected components ... is 4 > 1`**
Es el que importa: el grafo quedó partido. Ya explicado arriba.

**2. `SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive`**
Es ruido técnico, pero conviene entenderlo porque explica cómo se guardan los grafos.

Una **matriz dispersa** (*sparse*) es una matriz donde casi todo son ceros. El grafo de vecinos es así: con 342 pingüinos la matriz es de 342 × 342, unos 117.000 casilleros, pero cada punto se conecta solo con sus 15 vecinos, así que más del 95% son ceros. Guardar todos esos ceros sería un desperdicio, y por eso se guardan **solo los valores distintos de cero**, anotando su fila y su columna.

Hay varios formatos para organizar esa lista:

| Formato | Bueno para | Malo para |
|---|---|---|
| **CSR** (el que usa sklearn) | Leer y hacer cuentas rápido | **Insertar** valores nuevos |
| **LIL** o **DOK** | Insertar valores nuevos | Hacer cuentas |

CSR guarda los datos en tiras ordenadas y compactas, así que insertar un valor en el medio obliga a reacomodar todo lo que sigue, como insertar una fila en el medio de una planilla enorme.

**Por qué aparece:** para unir los bloques desconectados, sklearn **agrega conexiones** al grafo, es decir, inserta valores nuevos en una matriz CSR. Cada inserción dispara el aviso. La prueba de que es la misma causa es que con 30 vecinos, donde el grafo ya está conectado, **no aparece ninguno de los dos avisos**.

No indica ningún problema con los datos ni con los resultados: es una sugerencia de rendimiento dirigida a quien programó la librería.

### El error de reconstrucción

`iso.reconstruction_error()` mide cuánto se deforman las distancias al bajar de dimensión. Más chico es mejor, y baja al agregar componentes. Con nuestros datos y 11 vecinos: 4.32 con 1 componente, 1.28 con 2 y 0.67 con 3.

⚠️ **Solo se puede comparar entre distintos `n_components` con el mismo `n_neighbors`.** Al cambiar los vecinos cambia el grafo y, con él, las distancias geodésicas contra las que se mide el error: sería comparar contra dos varas distintas. **No sirve para elegir `n_neighbors`.**

### Diferencias con PCA

| | PCA | Isomap |
|---|---|---|
| Tipo | Lineal | No lineal |
| Qué preserva | La varianza (la dispersión global) | Las distancias geodésicas (la vecindad) |
| ¿Hay cargas interpretables? | Sí, `components_` dice qué variable pesa en cada eje | **No**: los ejes no tienen una receta en términos de las variables originales |
| ¿Varianza explicada? | Sí, permite elegir cuántas componentes | **No existe**; se usa el error de reconstrucción |
| ¿Se pueden proyectar datos nuevos? | Sí, fácil | Más costoso: depende del grafo |
| Parámetros | Ninguno relevante | `n_neighbors` cambia mucho el resultado |

**Para el parcial:** que los ejes de Isomap **no sean interpretables** es su gran desventaja frente a PCA. En PCA podíamos decir "PC1 es el tamaño"; en Isomap, ISO1 no significa nada concreto. A cambio, puede separar grupos que PCA no logra separar.

## 14. t-SNE

### La idea: vecindades en vez de distancias

PCA preserva la varianza. Isomap preserva las distancias geodésicas. **t-SNE preserva las vecindades**: no le importa que dos puntos queden a la distancia exacta, sino que **los que eran vecinos sigan siendo vecinos**.

Pensalo como armar una foto grupal: no importa que las distancias estén a escala, importa que cada uno quede al lado de los suyos.

### Cómo funciona, en 4 pasos

1. **Arma una tabla de vecindades en el espacio original.** Para cada par de pingüinos calcula una **probabilidad de ser vecinos**: alta si están cerca, casi cero si están lejos. Usa una campana de Gauss centrada en cada punto.
2. **Tira todos los puntos al azar en el plano.** De ahí viene su carácter aleatorio.
3. **Arma la misma tabla en el plano**, pero con una distribución t de Student, que tiene colas más pesadas y permite que los grupos distintos se separen más.
4. **Mueve los puntos de a poco**, iteración tras iteración, hasta que las dos tablas se parezcan lo más posible.

### El KL (divergencia de Kullback-Leibler)

Es la medida de **cuánto se diferencian las dos tablas** (la de vecindades en el espacio original y la del plano), y es lo que el algoritmo minimiza moviendo los puntos. `scikit-learn` lo deja en `tsne.kl_divergence_`. KL = 0 significa que el mapa reproduce las vecindades a la perfección; cuanto más chico, mejor copiado.

> ⚠️ **No se vio en clase.** El profesor objetó que lo usáramos en el TP1 para elegir la cantidad de componentes sin explicar qué es (sección 25). Sirve para entender qué optimiza t-SNE, pero **en una entrega no se justifica una decisión con una medida que no se vio**: las decisiones se toman con los criterios de clase y con lo que se observa en los gráficos.

**Cuándo se puede comparar:**

| Comparar KL entre… | ¿Vale? | Por qué |
|---|---|---|
| **Iteraciones** | **Sí** | La tabla original no cambia; menos KL = más convergido |
| **Componentes** | **Sí** | Con más dimensiones hay más lugar para acomodar los puntos |
| **Perplejidades** | **NO** | La perplejidad **define** la tabla original; se mediría contra otra vara |

Es la misma trampa que el error de reconstrucción de Isomap entre distintos `n_neighbors`. Con nuestros datos el KL baja de 0.49 (perplejidad 5) a 0.07 (perplejidad 100), y sería un error concluir que 100 es mejor: con perplejidad alta la referencia es más difusa y más fácil de copiar.

### Los parámetros

**`perplexity` (perplejidad).** Es el parámetro propio de t-SNE: una medida suave de **cuántos vecinos efectivos** considera cada punto. El rango recomendado es **5 a 50**, y siempre debe ser menor que la cantidad de puntos.

| Valor | Qué pasa | En nuestros datos (342 pingüinos) |
|---|---|---|
| Muy bajo (5) | Dominan las variaciones locales: los grupos se parten en fragmentos | Grupos despedazados |
| Intermedio (15) | Equilibrio | Seis grupos compactos y separados |
| Alto (30-50) | Los grupos se fusionan | Adelie y Chinstrap se pegan y terminan mezclándose |

**`max_iter` (iteraciones).** Cuántas veces mueve los puntos. Lo habitual es del orden de 1000. Si se corta antes de converger, aparecen formas raras y "pellizcadas", con grupos poco definidos. Eso se ve en el gráfico: con 300 iteraciones nuestros grupos quedaron extendidos y mal definidos, y entre 1000 y 2000 el dibujo es prácticamente el mismo (sin mirar el KL, que fue 0.549, 0.383, 0.361 y 0.353 para 300, 500, 1000 y 2000).

**Cómo se barren los dos parámetros.** Perplejidad e iteraciones se prueban de a uno, dejando el otro fijo, y al final se elige **la combinación**: una figura variando la perplejidad con las iteraciones fijas, otra variando las iteraciones con la perplejidad fija. En los dos casos hay que **decir en el texto cuál es el valor fijo**; ponerlo solo en el título de la figura no alcanzó (sección 25).

⚠️ En `scikit-learn` moderno el parámetro se llama **`max_iter`**; antes era `n_iter`. El mínimo admitido es 250, pero no debe usarse: ahí la optimización termina durante la fase inicial (*early exaggeration*) y el KL informado es un valor centinela gigantesco, no un error real.

**`n_components`.** Dimensiones de salida, 2 o 3 para visualizar.

**`random_state` (semilla).** Como arranca al azar, cada corrida da un dibujo distinto. Fijar la semilla hace el resultado reproducible. Con otra semilla **cambia el dibujo, no la calidad** (lo probamos con tres semillas y el KL casi no varió, aunque ese apartado no quedó en el TP). Por eso se recomienda correrlo varias veces y comparar.

### Qué NO hay que leer en un gráfico de t-SNE

Esto suele ser pregunta de parcial:

1. **Los ejes no significan nada.** No hay cargas ni interpretación posible, a diferencia de PCA.
2. **Los tamaños de los grupos no son reales.** El algoritmo expande las zonas densas y comprime las dispersas, así que todos los grupos terminan pareciendo del mismo tamaño.
3. **Las distancias entre grupos no son confiables.** Que dos grupos aparezcan lejos no significa que sean muy distintos.

Lo que sí se puede afirmar: **que los grupos existen y están diferenciados**.

### ⚠️ Cuidado con el KL en un informe

**Esto nos lo marcó el profesor en el TP1.** Usamos el KL para justificar decisiones, y la observación fue que **no se vio en clase**.

La regla que queda: **para justificar una decisión, usar solo criterios vistos en clase** (varianza acumulada, Kaiser, codo, Silhouette, GAP) y lo que se observa en los gráficos. Si una métrica no se vio, no sirve ni usarla ni explicarla, porque el enunciado además prohíbe incluir teoría en el informe.

El KL sigue siendo útil para **entender** qué está haciendo t-SNE, pero no para apoyar una elección en la entrega.

### Errores típicos con t-SNE

| Error | Por qué está mal |
|---|---|
| Leer las distancias entre grupos | t-SNE expande lo denso y comprime lo disperso: no son proporcionales a nada |
| Comparar tamaños de los grupos | Por lo mismo: todos tienden a parecer del mismo tamaño |
| Interpretar los ejes | `TSNE1` y `TSNE2` no significan nada |
| Correrlo una sola vez | Es aleatorio: conviene repetir y ver si la estructura se mantiene |
| Decir "proyección" | Proyectar es lo que hace PCA, que es lineal. En t-SNE e Isomap se dice "representación en dos dimensiones" |
| Usar perplejidad mayor que la cantidad de puntos | El algoritmo se comporta de forma imprevisible |

### Comparación de los tres métodos

| | PCA | Isomap | t-SNE |
|---|---|---|---|
| Tipo | Lineal | No lineal | No lineal |
| Qué preserva | La varianza (estructura global) | Distancias geodésicas (local y global) | Vecindades (estructura local) |
| ¿Ejes interpretables? | **Sí** (cargas) | No | No |
| ¿Determinista? | Sí | Sí | **No**, depende de la semilla |
| Medida de error | Varianza explicada | Error de reconstrucción | KL |
| Parámetro clave | Ninguno | `n_neighbors` | `perplexity` |
| Costo de cómputo | Bajo | Medio | Alto |
| En nuestros datos | No separa Adelie de Chinstrap en 2D | Las separa, contiguas | Las separa, con un contacto puntual |

**Regla práctica:** PCA para entender **qué** variables explican la variabilidad; t-SNE para **ver** si hay grupos; Isomap como intermedio cuando la estructura es curva.

## 15. Clustering y K-means

### Qué es el clustering

Hasta acá **reducíamos dimensiones** para poder *ver* los datos. El clustering hace otra cosa: **arma grupos** (clusters) de muestras parecidas entre sí, **sin mirar la etiqueta**. Es no supervisado: el algoritmo nunca ve la columna `Especie`. Después nosotros comparamos los grupos que armó con las especies, para ver si coinciden.

"Parecidas" significa **cercanas**: se mide con distancias, y por eso hay que estandarizar antes (si no, la masa en gramos decide todo).

### Cómo funciona K-means

Hay que decirle de antemano cuántos grupos queremos: ese número es **k**.

1. Ubica k **centroides** (el "centro" de cada grupo) en posiciones iniciales.
2. Asigna cada punto al centroide que tiene más cerca.
3. Mueve cada centroide al promedio de los puntos que le tocaron.
4. Repite 2 y 3 hasta que los grupos dejan de cambiar.

**Analogía:** k locales de una cadena que se van mudando hasta quedar cada uno en el medio de sus clientes.

El **centroide** es el pingüino "promedio" de su cluster: un valor por cada característica. En sklearn queda en `kmeans.cluster_centers_`, y el cluster de cada muestra en `kmeans.labels_`.

**`random_state=42`:** las posiciones iniciales tienen una parte de azar, así que dos corridas pueden dar grupos algo distintos (o los mismos con otro número). Fijar la semilla lo hace reproducible.

**Ojo:** los números de cluster (0, 1, 2...) son arbitrarios. "Cluster 0" no significa nada por sí mismo: hay que mirar qué muestras tiene adentro.

### La inercia y el método del codo

La **inercia** (`kmeans.inertia_`) es la suma de las distancias al cuadrado de cada punto a su centroide. Mide qué tan **apretados** están los grupos: cuanto menor, más compactos.

El problema es que **siempre baja al aumentar k** (con un cluster por punto sería 0), así que no se puede elegir "el k con menor inercia". Se grafica contra k y se busca el **codo**: el punto donde deja de bajar rápido. Igual que en PCA, **el k del quiebre se incluye**.

En nuestros datos: 1710 (k=1), 907, 593, 403, 301, 233, 217, 203... Lo que se mira es cuánto baja en cada paso: 803, 314, 190, 102, 68 y 16. El quiebre se lee **alrededor de k=4 o 5**, porque de ahí en adelante las caídas ya son chicas. No es un codo nítido, y por eso hacen falta índices más precisos.

> ⚠️ En el TP1 escribimos que la inercia "baja de forma pronunciada hasta k=6" (celda `km-04`). El profesor marcó que el quiebre está en 4 o 5, y aclaró que está bien pasar a otros criterios cuando el codo no alcanza (sección 25).

### El coeficiente de Silhouette

Para **cada punto** compara dos distancias:

- **a**: qué tan lejos está, en promedio, de los puntos de **su propio** cluster.
- **b**: qué tan lejos está, en promedio, de los puntos del cluster **vecino más cercano**.

Silhouette = (b − a) / max(a, b). Va de −1 a 1:

| Valor | Significado |
|---|---|
| Cerca de 1 | El punto está bien adentro de su cluster y lejos de los demás |
| Cerca de 0 | Está en la frontera entre dos clusters |
| Negativo | Probablemente quedó en el cluster equivocado |

`silhouette_score` devuelve el **promedio** de todos los puntos. Se calcula para varios k y se elige **el más alto**. Empieza en k=2 porque con un solo cluster no hay "vecino".

En nuestros datos: 0.443 (k=2), 0.450, 0.501, 0.513, **0.515 (k=6)**, 0.467, 0.412... El máximo es k=6, pero k=4 y k=5 quedan casi empatados.

#### Silhouette calculado a mano

Un punto cualquiera, con dos clusters:

```
  Cluster A (el suyo)        Cluster B (el vecino)
     ● ● ●                        ○ ○ ○
       ↑ nuestro punto
```

1. **a** = distancia promedio a los puntos **de su propio** cluster. Supongamos **2**.
2. **b** = distancia promedio a los puntos del cluster **vecino más cercano**. Supongamos **6**.
3. Se aplica la fórmula (filmina 42 de la unidad 3):

```
s = (b − a) / max(a, b) = (6 − 2) / 6 = 4/6 = 0.67
```

Un 0.67 indica que el punto está bastante mejor acompañado en su cluster que en el vecino.

**Los tres casos límite:**

| Situación | Cuentas | s | Interpretación |
|---|---|---|---|
| Muy cerca de los suyos, lejos del otro | a=1, b=9 | **0.89** | Bien clasificado |
| A igual distancia de ambos | a=5, b=5 | **0** | En el borde, ambiguo |
| Más cerca del cluster vecino que del propio | a=8, b=3 | **−0.63** | Probablemente mal asignado |

El `silhouette_score` del conjunto es el promedio de los `s` de todos los puntos. Nuestro máximo fue **0.515** con k=6.

### El estadístico GAP

**La idea:** comparar nuestros datos contra datos **sin ningún grupo**. Si K-means agrupa nuestros datos mucho mejor de lo que agrupa puntos tirados al azar, es que los grupos son reales.

Para cada k:

1. Se calcula la inercia de K-means sobre los datos reales.
2. Se generan varios conjuntos de puntos al azar, **uniformes dentro del mismo rango** que los datos (el "rectángulo" que los contiene), se calcula la inercia de cada uno y se promedia: es la **referencia**.
3. `GAP = log(inercia de referencia) − log(inercia real)`.

Un GAP alto significa que los datos reales quedan mucho más apretados que el azar. Se elige **el k con el GAP más alto**.

En nuestros datos: 0.35, 0.72, 1.00, 1.26, 1.44, **1.61 (k=6)**, 1.59, 1.56, 1.54, 1.60. Crece hasta 6 y después queda casi plano.

**Los dos errores del código de clase** (y por qué los corregimos):

1. La función hacía `kmeans.fit(X_std)` en lugar de `kmeans.fit(X)`: ignoraba los datos que recibía, así que la "referencia" era otra vez nuestros datos y el GAP no comparaba nada.
2. La referencia salía de `np.random.rand`, que da números entre 0 y 1. Nuestros datos estandarizados van de −2 a 3: la referencia quedaba amontonada en un rincón, su inercia era chiquita, el GAP daba **negativo** y elegía siempre k=10. La corrección es estirar esos números al rango de cada columna: `np.random.rand(...) * (máximo − mínimo) + mínimo`.

Además fijamos `np.random.seed(42)`, porque los puntos de referencia son al azar y sin semilla el GAP cambia en cada corrida.

### ⚠️ El codo en K-means: el k del quiebre se incluye

**El profesor también nos corrigió esto.** Dijimos que el quiebre estaba en k=6; su devolución fue que se produce en **k=4 o 5**.

Es la misma regla que en PCA: el codo es donde la curva **deja de caer bruscamente**, y ese k entra. Mirando nuestra tabla de inercia:

| k | Inercia | Cuánto bajó |
|---|---|---|
| 2 | 907.2 | — |
| 3 | 593.3 | −314 |
| 4 | 403.1 | −190 |
| 5 | 300.9 | −102 |
| 6 | 233.2 | −68 |
| 7 | 216.7 | −17 |

Las caídas se achican mucho entre k=4 y k=5: ahí está el codo. Aclaró además que **está bien pasar a otros criterios si el codo no alcanza**, que es lo que hicimos al elegir 6 por Silhouette y GAP.

### Qué nos dio: 6 clusters, no 3

Los dos índices coinciden en **k=6**, aunque por poco. No son las 3 especies: son las **6 combinaciones de especie y sexo**, los mismos seis grupos que se veían en Isomap y t-SNE.

| k | Qué arma K-means |
|---|---|
| 3 | Un cluster con todo Gentoo, uno con los **machos** de Adelie y Chinstrap, y uno con las **hembras** de Adelie y Chinstrap |
| 6 | Un cluster por especie y sexo; solo 5 pingüinos de 342 quedan fuera de su grupo |

**Por qué con k=3 no salen las especies:** K-means no sabe qué es una especie, solo ve distancias. Como metimos `Sexo` como característica, machos y hembras quedan a 2 unidades de distancia en esa dimensión, y "le conviene" cortar por sexo antes que separar Adelie de Chinstrap. Es la misma decisión que partió el grafo de Isomap en bloques.

**La tabla cruzada** (`pd.crosstab`) es la herramienta para leer un clustering: filas con lo que sabemos (especie y sexo), columnas con el cluster, y en cada celda cuántos pingüinos hay. Si cada fila cae casi entera en una sola columna, el clustering recuperó ese grupo.

### Límites de K-means

- Hay que elegir k de antemano.
- Arma grupos más o menos redondos; no sirve para formas raras.
- Es sensible a los valores atípicos, que arrastran a los centroides.
- Depende de las posiciones iniciales (por eso la semilla).

## 16. Clustering jerárquico

### La idea

K-means arma k grupos de una sola vez. El clustering jerárquico arma **un árbol** de grupos: muestra cómo se van juntando los datos, desde cada muestra por separado hasta un único grupo con todo. Después uno decide **a qué altura cortar** el árbol, y de eso sale la cantidad de clusters.

Hay dos tipos:

| Tipo | Cómo trabaja |
|---|---|
| **Aglomerativo** (el que usamos) | De abajo hacia arriba: empieza con cada muestra como un cluster y va **uniendo** los dos más parecidos |
| **Divisivo** | De arriba hacia abajo: empieza con todo junto y va **dividiendo** |

### El algoritmo aglomerativo, paso a paso

1. Cada pingüino es un cluster (342 clusters).
2. Se buscan los dos clusters más cercanos y se unen (quedan 341).
3. Se repite hasta que queda uno solo.

**Analogía:** un árbol genealógico al revés. Primero se juntan los hermanos, después los primos, después las familias, hasta llegar a un único antepasado.

### El enlace (`linkage`): cómo se mide la distancia entre dos grupos

Entre dos puntos la distancia es obvia. Entre dos **grupos** hay que elegir un criterio:

| Enlace | Distancia entre dos clusters | Característica |
|---|---|---|
| `single` | La de sus dos puntos **más cercanos** | Poco estable, arma cadenas alargadas |
| `complete` | La de sus dos puntos **más lejanos** | Grupos compactos |
| `average` | El **promedio** entre todos los pares | Intermedio |
| `centroid` | La de sus **centroides** | — |
| **`ward`** (el que usamos) | Une los dos clusters cuya fusión **menos aumenta la dispersión interna** | Grupos compactos y parejos; es el que usa clase |

`ward` busca lo mismo que K-means (grupos apretados alrededor de su centro), y por eso los dos métodos dieron resultados tan parecidos.

#### Un paso del algoritmo, con números

Cuatro puntos y sus distancias:

```
        B      C      D
  A    2.0    6.0    9.0
  B           5.0    8.0
  C                  3.0
```

1. **Primera fusión:** la distancia más chica es 2.0, entre A y B. Se unen en el grupo **{A,B}**.
2. **Recalcular distancias** del grupo nuevo a los demás. Acá importa el enlace:

| Enlace | Distancia de {A,B} a C | Criterio |
|---|---|---|
| `single` (mínimo) | min(6.0, 5.0) = **5.0** | El par más cercano |
| `complete` (máximo) | max(6.0, 5.0) = **6.0** | El par más lejano |
| `average` | (6.0 + 5.0)/2 = **5.5** | El promedio |

3. **Siguiente fusión:** ahora la menor es 3.0, entre C y D → grupo **{C,D}**.
4. Quedan dos grupos, que se unen al final.

El dendrograma dibuja exactamente esa secuencia, con la altura de cada unión igual a la distancia a la que ocurrió.

### El dendrograma y cómo leerlo

Es el dibujo del árbol.

- **Eje horizontal:** las muestras (o grupos de muestras, si está truncado).
- **Eje vertical:** la distancia a la que se unieron dos grupos.
- **Una unión baja** significa que esos grupos eran muy parecidos. **Una unión alta**, que eran muy distintos.

**Cómo se elige la cantidad de clusters:** se traza una línea horizontal y se cuentan las ramas verticales que corta. Conviene cortar donde hay un **tramo vertical largo sin uniones**, porque significa que los grupos de abajo son bien distintos entre sí.

En nuestro dendrograma las últimas uniones ocurren a alturas 6.3, **11.9**, 13.9, 19.5, 25.1 y 40.1. Entre 6.3 y 11.9 no pasa nada: cortando en cualquier altura de ese tramo (usamos 9) quedan **6 ramas**.

**Dendrograma truncado:** con 342 hojas el eje horizontal no se lee. `truncate_mode='lastp', p=20` muestra solo las últimas 20 ramas, y el número entre paréntesis es cuántas muestras hay debajo de cada una. Un número **sin** paréntesis es una muestra suelta (su índice), no una cantidad.

### En sklearn y scipy

Se usan dos herramientas distintas para lo mismo:

- `scipy`: `linkage(X_std, "ward")` arma el árbol completo y `dendrogram(Z)` lo dibuja.
- `sklearn`: `AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_std)` devuelve directamente el cluster de cada muestra para un k dado. Es como cortar el árbol para que queden k ramas.

**No lleva semilla:** a diferencia de K-means, no hay nada al azar. Siempre da el mismo resultado.

### Silhouette y GAP en jerárquico

Son los mismos índices que en K-means; solo cambia el método que arma los grupos.

| k | Silhouette | GAP |
|---|---|---|
| 5 | 0.511 | 1.488 |
| **6** | **0.515** | 1.677 |
| **7** | 0.465 | **1.684** |
| 8 | 0.409 | 1.663 |

Acá los índices **no coinciden**: Silhouette dice 6 y GAP dice 7. Pero la diferencia del GAP entre 6 y 7 es de 0.007, es decir, nada. Mirando qué hace el séptimo cluster, solo parte en dos a los machos de Gentoo, una división que no corresponde a nada conocido. Por eso concluimos que **6 representa mejor los datos**, aunque el óptimo "por GAP" sea 7.

**La lección:** un índice no decide solo. Cuando la curva es plana, el máximo es casi una casualidad, y hay que mirar los otros índices y qué grupos se forman.

**El error del código de clase:** para calcular la dispersión restaba `X_np[labels] - centroids[labels]`. `X_np[labels]` no es "los puntos": usa las etiquetas (0, 1, 2...) como números de fila, así que agarra siempre las primeras filas de la tabla. Lo correcto es `X_np - centroids[labels]`: a cada punto, restarle el centroide de **su** cluster.

### Jerárquico contra K-means

| | K-means | Jerárquico (ward) |
|---|---|---|
| Hay que dar k de antemano | Sí | No: se decide al cortar el árbol |
| Resultado | Una partición | Un árbol con todos los niveles |
| Depende del azar | Sí (semilla) | No |
| Costo con muchos datos | Bajo | Alto |
| En nuestros datos | 6 clusters, 5 pingüinos fuera de su grupo | 6 clusters, 5 pingüinos fuera de su grupo |

Los dos llegaron a los mismos seis grupos (especie × sexo). Los 5 que quedan fuera no son exactamente los mismos pingüinos, pero en ambos casos son de Adelie o de Chinstrap, las dos especies más parecidas.

## 17. Cómo funciona scikit-learn

Todos los métodos se usan igual, y por eso conviene entender el patrón una sola vez:

| Método | Qué hace |
|---|---|
| **`fit(X)`** | **Aprende** los parámetros a partir de los datos (la media y el desvío de cada columna; las direcciones de PCA) |
| **`transform(X)`** | **Aplica** lo aprendido y devuelve los datos transformados |
| **`fit_transform(X)`** | Las dos cosas de una |
| **`fit_predict(X)`** | Aprende y devuelve una etiqueta por muestra (lo usan los métodos de clustering) |

**Atributos aprendidos:** los que terminan en guion bajo existen recién **después** del `fit`: `scaler.mean_`, `pca.explained_variance_`, `pca.components_`.

**`set_output(transform="pandas")`:** hace que el resultado sea un DataFrame con nombres de columna e índice, en vez de un arreglo de numpy pelado. Conviene siempre: permite graficar con nombres y mantiene cada fila alineada con `y`.

**`Pipeline`:** encadena varios pasos (por ejemplo escalar y después PCA) en un solo objeto, para no olvidarse ninguno y poder repetir el proceso igual. Nosotros no lo usamos en PCA porque ya teníamos los datos estandarizados de la consigna anterior.

**Un detalle que confunde:** sklearn numera las columnas de salida **desde 0**. `pca0` es la primera componente (PC1), `pca1` la segunda, y así. En nuestro notebook las renombramos apenas se calculan:

```python
X_pca.columns = [f"PC{i+1}" for i in range(pca.n_components_)]
```

Así el nombre de la columna coincide con el número de componente y los gráficos quedan rotulados PC1, PC2 y PC3. Es la misma confusión que hizo que el notebook de clase graficara en 3D las componentes 2, 3 y 4 creyendo que eran las tres primeras.

---

## 18. Aprendizaje supervisado

### La diferencia con lo que veníamos haciendo

Imaginen dos formas de enseñarle frutas a un chico:

1. **Le mostramos frutas diciéndole el nombre de cada una.** "Esto es una manzana, esto es una pera, esto es una manzana..." Después le damos una fruta nueva y le preguntamos qué es. → **Supervisado**: aprendió con las respuestas a la vista.
2. **Le damos una bolsa de frutas mezcladas y le pedimos que arme montoncitos** con las que se parecen entre sí. Nunca le dijimos cómo se llaman. → **No supervisado**: encontró estructura por su cuenta.

El TP1 era el caso 2: PCA, Isomap, t-SNE y los clustering nunca vieron la especie del pingüino. El TP2 es el caso 1.

| | No supervisado (TP1) | Supervisado (TP2) |
|---|---|---|
| ¿El modelo ve la respuesta? | **Nunca** | **Sí, mientras entrena** |
| Qué busca | Estructura escondida | Una regla para predecir |
| Cómo se evalúa | Índices internos (Silhouette) | Comparando predicción contra realidad |
| Ejemplos | PCA, K-means, jerárquico | Árboles, Naive Bayes, k-NN, SVM |

### El vocabulario, que conviene tener clarísimo

| Término | Qué significa | En nuestro TP2 |
|---|---|---|
| **Característica** (*feature*), predictor, variable de entrada | Lo que el modelo usa para decidir | `RyD_Spend`, `Edad`, `BP`... |
| **Variable objetivo** (*target*), clase, etiqueta | Lo que hay que predecir | `Profit` (Parte A), `Droga` (Parte B) |
| **`X`** | La tabla con todas las características | 996 × 6 en la Parte A |
| **`y`** | La columna con la respuesta | 996 valores de ganancia |
| **Muestra** / instancia / observación | Una fila | Una empresa, un paciente |
| **Modelo** | La regla aprendida | El árbol con sus preguntas y umbrales |
| **Entrenar** (`fit`) | Que el modelo mire `X` e `y` y arme la regla | — |
| **Predecir** (`predict`) | Aplicar la regla a datos nuevos | — |

### Parámetros contra hiperparámetros

Esta distinción suele ser pregunta de parcial.

| | **Parámetros** | **Hiperparámetros** |
|---|---|---|
| Quién los fija | **El algoritmo**, solo, al entrenar | **Nosotros**, antes de entrenar |
| Ejemplo en un árbol | Qué variable usa cada nodo y en qué umbral corta | `max_depth`, `min_samples_leaf` |
| Ejemplo en k-NN | (no tiene: guarda los datos) | `n_neighbors`, `metric` |
| Cómo se eligen | Se aprenden de los datos | Probando combinaciones (`GridSearchCV`) |

Metáfora: los hiperparámetros son **las perillas del horno** (temperatura, tiempo) que uno elige antes de cocinar; los parámetros son **cómo queda la torta por dentro**, que surge del proceso.

### Los dos tipos de problema supervisado

| | **Regresión** | **Clasificación** |
|---|---|---|
| Qué predice | Un **número** | Una **categoría** |
| Nuestro caso | `Profit`: 118.675, 92.300... | `Droga`: una de 5 etiquetas |
| ¿Tiene sentido promediar la respuesta? | Sí | No |
| ¿Tiene sentido decir "se equivocó por 300"? | Sí | No: o acierta o no |
| Métricas | MAE, MSE, RMSE | Exactitud, precisión, exhaustividad |

**Cómo reconocerlo en un enunciado:** si la respuesta es un número que podría tomar infinitos valores, es regresión. Si es una de unas pocas opciones, es clasificación. Ojo con las trampas: predecir "cuántos hijos tiene una familia" (0, 1, 2, 3) es regresión aunque sean números enteros; predecir "nivel: bajo, medio, alto" es clasificación aunque exista un orden.

### El flujo completo de un trabajo supervisado

```
1. Cargar y explorar los datos           (EDA)
2. Limpiar                                (nulos, duplicados, outliers)
3. SEPARAR en entrenamiento y test        ← el paso que cambia todo
4. Preparar, aprendiendo solo del train   (codificar, escalar)
5. Elegir hiperparámetros con el train    (validación cruzada)
6. Entrenar el modelo final con el train
7. Evaluar UNA SOLA VEZ con el test
```

Los pasos 3 a 7 son nuevos del TP2. El orden **no es negociable**: cada vez que se adelanta el paso 4 antes del 3, aparece la fuga de datos (sección 20).

---

## 19. Entrenamiento y test

### El problema, con una metáfora

Un profesor le da a un alumno 100 ejercicios resueltos para estudiar. Llega el examen y le toma **esos mismos 100 ejercicios**.

El alumno que entendió saca 9. Pero el que se aprendió las respuestas de memoria, sin entender nada, saca 10. El examen no sirvió para distinguirlos.

Eso pasa si entrenamos y evaluamos con los mismos datos: **el modelo que memoriza gana siempre**, y no es el que queremos. Lo que nos interesa es que funcione con una empresa que nunca vio.

A esa capacidad se la llama **generalización**, y es el concepto central de todo el aprendizaje supervisado.

### La solución

Antes de entrenar, se apartan algunas filas y se guardan bajo llave:

```
   996 empresas
        │
        ├──── 796 (80%) → ENTRENAMIENTO: el modelo aprende acá
        │
        └──── 200 (20%) → TEST: guardado, se usa una sola vez al final
```

**El test no se toca para nada más.** No se mira para elegir hiperparámetros, no se usa para decidir qué variables sacar, no se usa para ajustar el escalador. Es el examen final.

### `train_test_split`, parámetro por parámetro

```python
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2,
                                          random_state=63, shuffle=True)
```

| Parámetro | Qué hace | Qué pasa si se omite |
|---|---|---|
| `test_size=0.2` | Reserva el 20% para test | Por defecto 0.25 |
| `random_state=63` | Fija el sorteo de qué filas van a cada lado | **Cada corrida da filas distintas**, y los resultados cambian: imposible reproducir el informe |
| `shuffle=True` | Mezcla antes de cortar (viene activado) | Si el archivo está ordenado (por ejemplo, por clase), el test podría quedar con una sola categoría |
| `stratify=y` | Conserva la proporción de clases en ambos lados | Con clases raras, pueden quedar casi todas de un lado |

**Qué es `random_state` en criollo:** la computadora no sabe sortear de verdad; usa una fórmula que produce números que *parecen* al azar. La semilla es el punto de partida de esa fórmula. Con la misma semilla, el mismo "sorteo". Es como barajar un mazo siempre con el mismo movimiento exacto: parece desordenado, pero da siempre el mismo orden.

Por eso el enunciado pide `random_state=63` en todo: para que el corrector obtenga exactamente nuestros números.

### Devuelve cuatro cosas, en ese orden

```python
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, ...)
#  │     │     │     └── respuestas del test (se comparan al final)
#  │     │     └──────── respuestas del entrenamiento (el modelo las ve)
#  │     └────────────── características del test
#  └──────────────────── características del entrenamiento
```

Un error clásico es confundir el orden y recibir `Xtrn, ytrn, Xtst, ytst`. Siempre es **primero las dos X, después las dos y**.

### Estratificación, con números

`Droga` tiene 5 clases muy desparejas: DrogaY 91, DrogaX 54, DrogaA 23, DrogaC 16, DrogaB 16.

Si el test es el 20% (40 pacientes) y se sortea sin estratificar, podría tocar que **ninguno** sea de DrogaB. Entonces:
- No se puede medir si el modelo reconoce esa clase.
- Y si por azar quedan todas en entrenamiento, el modelo la aprende pero nunca se lo evalúa.

Con `stratify=y`, sklearn sortea **dentro de cada clase por separado** y arma el test respetando las proporciones: de 16 DrogaB, unas 3 van al test.

**Regla:** en clasificación, estratificar siempre. En regresión no aplica (no hay clases que mantener).

### Por qué el TP pide dos particiones

El enunciado pide 80/20 **y** 70/30 para ver si las conclusiones aguantan al cambiar cuánto se entrena.

| | Entrena con | Test de | Consecuencia |
|---|---|---|---|
| 80/20 | Más datos | 200 filas | Mejor modelo, medición más inestable |
| 70/30 | Menos datos | 299 filas | Modelo quizá peor, medición más confiable |

Es un compromiso: cuanto más grande el test, más confiable la medición, pero menos datos quedan para aprender.

**Cuidado al interpretar:** si una partición da mejores métricas, eso **no** significa que sea "la mejor partición". Son dos sorteos distintos: con un test chico, un par de casos difíciles alcanza para mover el promedio.

### El desbalance de clases

Una clase está **desbalanceada** cuando aparece mucho más o mucho menos que las otras. En `drugType.csv`:

| Droga | Pacientes | Porcentaje |
|---|---|---|
| DrogaY | 91 | **45.5%** |
| DrogaX | 54 | 27.0% |
| DrogaA | 23 | 11.5% |
| DrogaC | 16 | **8.0%** |
| DrogaB | 16 | **8.0%** |

**Por qué importa.** Un modelo que prediga **siempre DrogaY**, sin mirar nada, acierta el 45.5% de las veces. Si solo miramos la exactitud, ese modelo inútil parece medio decente. Con desbalances extremos es peor: en un problema donde el 99% es "sano", predecir siempre "sano" da 99% de exactitud y no detecta ningún enfermo.

**Las tres consecuencias prácticas:**

1. **Estratificar las particiones** (`stratify=y`), para que las clases chicas aparezcan en ambos conjuntos.
2. **No mirar solo la exactitud**: hay que ver el rendimiento **por clase** (precisión y exhaustividad, sección 24) y la matriz de confusión.
3. **Reportar en macro**: el promedio simple entre clases, sin pesar por tamaño, para que DrogaB (8%) pese lo mismo que DrogaY (45%).

Otra opción, que usa el notebook de la unidad 4, es `class_weight='balanced'`: hace que el modelo le dé más peso a los errores en las clases chicas durante el entrenamiento.

### Cómo se comprueba que la estratificación funcionó

Comparando el porcentaje de cada clase entre el conjunto completo y cada partición. Con nuestros datos:

| Droga | Conjunto completo | Train 80/20 | Test 80/20 |
|---|---|---|---|
| DrogaY | 45.5% | 45.6% | 45.0% |
| DrogaB | 8.0% | 8.1% | 7.5% |

Las proporciones se mantienen. Sin estratificar, con 16 pacientes de DrogaB y un test de 40, podría haber quedado ninguno.

### Cuando aparece un tercer conjunto: validación

Algunos notebooks de la cátedra parten en **tres**:

```
Datos → ENTRENAMIENTO ──→ el modelo aprende
      → VALIDACIÓN   ──→ se prueban hiperparámetros acá
      → TEST         ──→ medición final, una sola vez
```

La idea es la misma: si uno elige hiperparámetros mirando el test, el test se contamina. La validación cruzada (sección 21) resuelve lo mismo sin tener que apartar un tercer conjunto.

---

## 20. Fuga de datos (*data leakage*)

**Es el error conceptual más grave del TP2**, y la filmina 70 de la unidad 4 lo marca especialmente.

### La metáfora

Un profesor prepara el examen. Para que sea "representativo", calcula el promedio de notas de todo el curso... **incluyendo las notas del examen que todavía no tomó**. Imposible, claro. Pero eso es exactamente lo que hacemos si calculamos la media para estandarizar usando también las filas del test.

Otra forma de verlo: el test tiene que ser **una caja cerrada**. Todo lo que hagamos antes de abrirla debe poder hacerse sin espiar adentro.

### Qué significa "ajustar" una transformación

Varias transformaciones **aprenden números de los datos**:

| Transformación | Qué aprende | Dónde lo guarda |
|---|---|---|
| `StandardScaler` | La media y el desvío de cada columna | `scaler.mean_`, `scaler.scale_` |
| `OneHotEncoder` / `OrdinalEncoder` | Qué categorías existen | `enc.categories_` |
| `SimpleImputer` | El valor con el que rellena | `imputer.statistics_` |
| `Winsorizer` | Los límites del IQR | — |
| `KBinsDiscretizer` | Dónde cortar los intervalos | `est.bin_edges_` |

Si cualquiera de esos números se calcula con los datos del test, el test dejó de ser una caja cerrada.

### La regla, y cómo se escribe

```python
# 1. PRIMERO separar
Xtrn, Xtst, ytrn, ytst = train_test_split(X, y, test_size=0.2, random_state=63)

# 2. El escalador APRENDE solo del entrenamiento
scaler = StandardScaler()
scaler.fit(Xtrn)            # ← acá mira los datos y calcula media y desvío

# 3. Se APLICA a los dos
Xtrn_std = scaler.transform(Xtrn)
Xtst_std = scaler.transform(Xtst)   # ← usa la media del TRAIN, no la suya
```

⚠️ **`fit_transform` sobre el test es el error clásico.** `fit_transform(Xtst)` recalcularía la media con el test. Sobre el test va **siempre** `transform` a secas.

### Cómo se detecta

**1. Mirando el test estandarizado.** Si está bien hecho, el test **no** queda con media 0 y desvío 1. Esta es la tabla real de nuestro TP2:

| | media entrenamiento | desvío entrenamiento | media test | desvío test |
|---|---|---|---|---|
| RyD_Spend | 0.0 | 1.001 | **0.086** | **1.047** |
| Administration | 0.0 | 1.001 | **0.007** | **1.138** |
| MarketingSpend | 0.0 | 1.001 | **0.097** | **1.057** |

Si las cuatro columnas dieran 0 y 1, el escalador se ajustó con todo junto: fuga.

**2. Métricas demasiado buenas.** Un modelo que en test da casi perfecto suele ser señal de que algo se filtró.

### Las otras formas de fuga

| Forma | Qué pasa | Cómo se evita |
|---|---|---|
| **Duplicados repartidos** | Una fila idéntica queda en train y en test: el modelo es evaluado con algo que ya vio | Eliminarlos **antes** de partir |
| **Elegir hiperparámetros con el test** | Probar 36 combinaciones y quedarse con la mejor *en test* convierte al test en parte del entrenamiento | Validación cruzada (sección 21) |
| **Elegir variables con el test** | Descartar columnas mirando su correlación calculada sobre todos los datos | Decidirlo con el entrenamiento |
| **Datos temporales mal partidos** | Entrenar con datos de mañana para predecir lo de ayer | Partir por fecha, no al azar |

### Por qué importa tanto

Porque **la fuga no da error**. El código corre, las métricas salen y encima salen mejores. El problema aparece recién cuando el modelo se usa en la realidad y rinde mucho peor de lo prometido.

---

## 21. Validación cruzada y GridSearchCV

### El problema

Hay que elegir hiperparámetros. ¿Profundidad 3, 5 o 10? Para saberlo hay que probar y comparar. Pero:

- Si comparamos **en entrenamiento**, gana siempre el árbol más grande: memoriza mejor.
- Si comparamos **en test**, lo contaminamos y deja de ser una medición imparcial.

Hace falta un tercer lugar donde probar.

### La solución: k-fold

Se parte **el entrenamiento** en k pedazos iguales (*folds*). Con k = 5:

```
Entrenamiento: [ A | B | C | D | E ]

Vuelta 1:  entrena con B C D E   → evalúa con A
Vuelta 2:  entrena con A C D E   → evalúa con B
Vuelta 3:  entrena con A B D E   → evalúa con C
Vuelta 4:  entrena con A B C E   → evalúa con D
Vuelta 5:  entrena con A B C D   → evalúa con E

Resultado = promedio de las 5 evaluaciones
```

**Metáfora:** en vez de un simulacro de examen, se hacen cinco, rotando qué parte de los apuntes se tapa. El promedio dice mucho más que un único simulacro, donde podría haber tocado justo el tema que uno sabía.

**Ventajas:** cada fila se usa para evaluar exactamente una vez, se aprovechan todos los datos y la estimación es más estable. **Costo:** entrenar k veces en vez de una.

**Por qué 5 o 10:** la filmina 43 de la unidad 4 lo dice directamente: "K típicamente 5 o 10". Con k muy chico, cada evaluación usa pocos datos; con k muy grande (k = n, llamado *leave-one-out*), hay que entrenar n veces.

**Versión estratificada:** en clasificación, sklearn usa automáticamente `StratifiedKFold`, que arma los folds respetando la proporción de clases.

### GridSearchCV: la grilla

*Grid* quiere decir grilla o rejilla: todas las combinaciones posibles de los valores que uno propone.

```python
grilla = {"max_depth": [3, 5, 10, None],        # 4 valores
          "min_samples_leaf": [1, 5, 10],        # 3 valores
          "min_samples_split": [2, 5, 10]}       # 3 valores
```

Eso son 4 × 3 × 3 = **36 combinaciones**. Con `cv=5`, cada una se entrena 5 veces: **180 ajustes** en total. Por eso la búsqueda tarda bastante más que entrenar un solo modelo.

```python
busqueda = GridSearchCV(DecisionTreeRegressor(random_state=63),
                        grilla, cv=5, scoring="neg_mean_squared_error")
busqueda.fit(Xtrn, ytrn)      # ← solo el entrenamiento
```

Al terminar, `GridSearchCV` **reentrena la mejor combinación con todo el entrenamiento** y la deja lista en `best_estimator_`.

| Atributo | Qué devuelve |
|---|---|
| `best_params_` | El diccionario con la combinación ganadora |
| `best_estimator_` | El modelo ya entrenado, listo para `predict` |
| `best_score_` | El puntaje promedio de la ganadora en validación cruzada |
| `cv_results_` | La tabla completa de las 36 combinaciones, para justificar |

### Los puntajes negativos

sklearn tiene una regla única: **un score siempre se maximiza, más alto es mejor**. Con la exactitud funciona (0.95 > 0.80). Con los errores no, porque ahí más bajo es mejor.

Para no tener dos reglas, sklearn le cambia el signo a los errores. Por eso existe `neg_mean_squared_error`: es el MSE multiplicado por −1.

| Combinación | Puntaje de sklearn | MSE real |
|---|---|---|
| Peor | −30.244.068 | 30.244.068 |
| **Mejor** | **−5.020.219** | **5.020.219** |

−5.020.219 es mayor que −30.244.068 (está más cerca del cero), así que "elegir el más alto" sigue funcionando. Para leerlo hay que darlo vuelta:

```python
np.sqrt(-busqueda.best_score_)   # el - lo pasa a positivo, la raíz lo lleva a RMSE
```

### Los empates

Es normal que varias combinaciones den exactamente el mismo puntaje. Suele pasar cuando un hiperparámetro **no llega a actuar**: si el árbol deja de crecer solo en profundidad 7, poner `max_depth=10` o no poner límite da el mismo árbol.

Que los primeros puestos estén muy cerca (en nuestro caso, 2.241 contra 2.261, menos del 1%) dice algo útil: **la elección de hiperparámetros no es crítica en ese problema**. Conviene escribirlo en el informe.

### El esquema completo

```
       ┌──────────────── DATOS ────────────────┐
       │                                        │
   ENTRENAMIENTO (80%)                      TEST (20%)
       │                                        │
   GridSearchCV:                            guardado
   corta en 5 folds, prueba                 bajo llave
   36 combinaciones, elige                      │
       │                                        │
   mejor modelo ────────── predict ────────────→│
                                                 │
                                          MAE / MSE / RMSE
```

---

## 22. Árboles de decisión

### La idea

Un árbol de decisión es el juego de las 20 preguntas. Para estimar la ganancia de una empresa:

> ¿Gasta en I+D menos de 82.150? → **Sí** → ¿Menos de 40.895? → **Sí** → ¿Menos de 20.963? → **Sí** → *la ganancia estimada es 57.695*

Cada pregunta parte los datos en dos. Al final de cada camino hay una respuesta.

```
                  ¿RyD_Spend <= 82.150?
                   /                  \
                 sí                    no
                 /                      \
     ¿RyD <= 40.895?              ¿RyD <= 119.900?
       /        \                    /         \
    ...        ...                 ...         ...
```

### El vocabulario del árbol

| Término | Qué es |
|---|---|
| **Nodo raíz** | La primera pregunta, arriba de todo. Contiene todas las muestras |
| **Nodo interno** | Cualquier pregunta intermedia |
| **Rama** | Cada salida de una pregunta (sí / no) |
| **Hoja** | El final del camino. No pregunta nada: da la respuesta |
| **Profundidad** | La cantidad de preguntas del camino más largo |
| **División** (*split*) | El acto de partir un nodo en dos |

**Qué responde una hoja:**
- **Regresión:** el **promedio** de las muestras que cayeron ahí.
- **Clasificación:** la clase **mayoritaria** de las que cayeron ahí.

### Cómo elige cada pregunta (lo importante)

El árbol prueba **todos los cortes posibles de todas las variables** y se queda con el que deja los dos grupos resultantes lo más **homogéneos** posible. Homogéneo significa "parecidos entre sí en la respuesta".

Para medir eso hay dos familias de medidas según el problema.

#### En clasificación: Gini y entropía

Las dos miden **impureza**: cuán mezclado está un nodo.

- **Nodo puro** (todos de la misma clase) → impureza **0**.
- **Nodo mitad y mitad** → impureza **máxima**.

**Gini** se calcula como `1 − Σ pᵢ²`, donde `pᵢ` es la proporción de cada clase.

*Ejemplo hecho a mano.* Un nodo con 10 pacientes: 6 de DrogaY y 4 de DrogaX.

```
p(Y) = 6/10 = 0.6        p(X) = 4/10 = 0.4
Gini = 1 − (0.6² + 0.4²) = 1 − (0.36 + 0.16) = 1 − 0.52 = 0.48
```

Si el nodo fuera 10 de DrogaY y 0 de DrogaX:

```
Gini = 1 − (1² + 0²) = 1 − 1 = 0      ← nodo puro
```

**Entropía** mide lo mismo con otra fórmula (`−Σ pᵢ·log₂(pᵢ)`) y va de 0 a 1 en el caso de dos clases. Gini es el criterio por defecto de sklearn; en la práctica dan árboles muy parecidos, y Gini es un poco más rápido de calcular.

#### Cómo se compara un corte: la función de costo

No alcanza con mirar un nodo hijo: hay que mirar los dos, **pesados por cuántas muestras tiene cada uno**.

*Ejemplo de las filminas (pág. 24-28), con 20 personajes, 10 "malos" y 10 "buenos":*

| Corte | Hijo izquierdo | Hijo derecho | Costo |
|---|---|---|---|
| **A** | 4 rojos, 0 verdes (Gini 0) | 6 rojos, 10 verdes (Gini 0.47) | 0.375 |
| **B** | 2 rojos, 0 verdes (Gini 0) | 8 rojos, 10 verdes (Gini 0.49) | 0.446 |

El costo sale de promediar las impurezas pesando por el tamaño: `(4/20)×0 + (16/20)×0.47 = 0.375`. **Gana el corte A**, que tiene menor costo.

El árbol repite esto en cada nodo, con todos los cortes posibles. Ese procedimiento se llama **CART**.

#### En regresión: el error cuadrático

No hay clases, así que "homogéneo" significa **respuestas parecidas entre sí**. Se mide con el error cuadrático medio dentro del nodo: cuánto se alejan los valores de su propio promedio.

En la Figura 4 de nuestro TP2 esto se ve numéricamente: el `squared_error` baja de 1.555 millones en la raíz a 416 y 352 millones en el nivel siguiente. Cada corte deja grupos con ganancias más parecidas.

### De dónde sale un umbral como `RyD_Spend <= 82150.5`

Una duda habitual: ¿cómo "sabe" el modelo qué pregunta hacer? **No lo sabe: las prueba todas y se queda con la mejor.** Es fuerza bruta.

#### Paso 1: armar la lista de preguntas candidatas

Para cada variable, ordena sus valores y toma el **punto medio entre cada par consecutivo**. Si dos empresas gastaron 81.836 y 82.465:

```
(81.836 + 82.465) / 2 = 82.150,5     ← un candidato
```

Por eso los umbrales terminan en ,5 o en decimales raros: son puntos medios, no valores que existan en los datos. Se parte por el medio para que el corte quede lo más lejos posible de ambas muestras.

En nuestro nodo raíz, con 796 empresas y 6 columnas, salieron **2.382 cortes candidatos**.

#### Paso 2: puntuar cada candidato

Se parte el nodo y se mide cuán desparejos quedan los dos hijos, **pesando por cuántas muestras tiene cada uno**:

```
costo = (n_izq × impureza_izq + n_der × impureza_der) / n_total
```

La impureza es el error cuadrático (regresión) o Gini / entropía (clasificación).

#### Paso 3: quedarse con el menor costo

La cuenta real del nodo raíz de nuestro TP2:

| | Empresas | Ganancia media | MSE |
|---|---|---|---|
| Izquierda (≤ 82.150,5) | 414 | 85.078 | 416.126.989 |
| Derecha (> 82.150,5) | 382 | 153.531 | 352.183.105 |

```
costo = (414 × 416.126.989 + 382 × 352.183.105) / 796 = 385.440.351
```

El nodo **sin partir** tenía un MSE de 1.555.005.088. El corte lo baja a la cuarta parte: por eso vale la pena.

**Comparación con otros cortes de la misma variable:**

| Corte | Costo |
|---|---|
| **82.150,5** (el elegido) | **385.440.351** |
| 82.618 | 385.463.508 |
| 81.479 | 385.494.797 |
| 120.000 (arbitrario) | 634.376.639 |

Los umbrales cercanos dan costos casi iguales (el corte "no es mágico", hay toda una zona buena), pero uno lejano empeora mucho.

**Comparación entre variables** (el mejor corte de cada una):

| Variable | Su mejor umbral | Costo |
|---|---|---|
| **RyD_Spend** | 82.150,5 | **385.440.351** |
| MarketingSpend | 225.993 | 410.041.306 |
| Administration | 122.490 | 459.791.980 |
| State_Florida | 0,5 | 1.550.411.063 |

Las columnas de `State` casi no reducen el costo respecto del nodo sin partir (1.555 millones). Por eso terminaron con importancia **0**: el árbol nunca las eligió.

#### Paso 4: repetir en cada hijo

El mismo procedimiento se aplica a los 414 de la izquierda y a los 382 de la derecha, y así sucesivamente, hasta que un hiperparámetro lo frene (`max_depth`, `min_samples_leaf`...) o el nodo quede puro.

#### Una propiedad importante: es un algoritmo voraz (*greedy*)

Elige el mejor corte **en cada momento**, sin evaluar si uno peor ahora permitiría cortes mucho mejores más abajo. Es como jugar al ajedrez mirando solo la jugada siguiente.

Consecuencia: **el árbol que se obtiene no es necesariamente el mejor árbol posible**, sino el que resulta de ir eligiendo bien paso a paso. Encontrar el óptimo global sería computacionalmente inviable.

#### Y con las variables categóricas

Para una columna binaria (0/1), como las de one-hot, el único corte posible es `<= 0.5`, que equivale a preguntar "¿es de este estado, sí o no?". Por eso en la tabla de arriba el umbral de `State_Florida` es 0,5.

### Cómo se lee un nodo en `plot_tree`

Cada cajita tiene cuatro líneas:

```
┌──────────────────────────────┐
│ RyD_Spend <= 82150.5         │  ← la pregunta (las hojas no la tienen)
│ squared_error = 1555005087   │  ← impureza: qué tan dispares son las respuestas
│ samples = 796                │  ← cuántas muestras llegaron
│ value = 117928.4             │  ← la respuesta del nodo (promedio)
└──────────────────────────────┘
```

**Izquierda siempre es "sí"** (se cumple la condición), derecha es "no". En clasificación, en lugar de `squared_error` aparece `gini`, y `value` muestra el recuento por clase, más una línea `class =` con la mayoritaria.

El **color** indica el valor predicho: más intenso, más alto (regresión) o más puro (clasificación).

### Sobreajuste y subajuste

| | Qué pasa | Error en entrenamiento | Error en test |
|---|---|---|---|
| **Subajuste** | El árbol es tan simple que no capta el patrón | Alto | Alto |
| **Justo** | Capta el patrón, ignora el ruido | Bajo | Bajo |
| **Sobreajuste** | Memoriza hasta el ruido | **Casi 0** | Alto |

Un árbol sin límites sigue partiendo hasta dejar **una hoja por muestra**: ahí acierta el 100% en entrenamiento, pero aprendió las particularidades de esas filas, no la regla general.

**Metáfora del alumno:** el que subajusta estudió solo los títulos; el que sobreajusta se aprendió los ejercicios de memoria, palabra por palabra, y se cuelga si cambian un número.

#### Sesgo y varianza

Es la forma técnica de decir lo mismo, con la metáfora del tiro al blanco:

| | Qué es | Dardos |
|---|---|---|
| **Sesgo** alto | El modelo simplifica demasiado | Todos juntos, pero lejos del centro |
| **Varianza** alta | El modelo es hipersensible a los datos | Desparramados por todo el tablero |

Subajuste = mucho sesgo. Sobreajuste = mucha varianza. Hay que buscar el punto medio, y en árboles eso se controla con los hiperparámetros y la poda.

### Los hiperparámetros, uno por uno

| Parámetro | Qué controla | Valor chico | Valor grande |
|---|---|---|---|
| `max_depth` | Cuántas preguntas encadena como máximo | Árbol simple: riesgo de subajuste | Árbol profundo: riesgo de sobreajuste |
| `min_samples_leaf` | Mínimo de muestras que debe quedar **en una hoja** | Hojas con 1 caso: sobreajuste | Hojas grandes: árbol simple |
| `min_samples_split` | Mínimo de muestras para **intentar partir** un nodo | Parte casi siempre | Corta antes: árbol simple |
| `criterion` | La medida de impureza | `gini` (defecto) o `entropy` | — |
| `ccp_alpha` | Poda posterior | 0: no poda nada | Poda agresiva |
| `class_weight` | Compensar clases desbalanceadas | — | `'balanced'` da más peso a las raras |

**La diferencia entre `min_samples_leaf` y `min_samples_split`**, que se confunden: el primero mira **el resultado** (ninguna hoja puede quedar con menos de ese número); el segundo mira **antes de partir** (un nodo con menos de ese número ni lo intenta).

### Prepoda y poscada (poda)

| | Cuándo actúa | Cómo |
|---|---|---|
| **Prepoda** | Mientras crece | `max_depth`, `min_samples_leaf`, `min_samples_split` |
| **Poda** (*post-pruning*) | Después, sobre el árbol ya crecido | `ccp_alpha` |

La poda deja crecer el árbol entero y después **corta las ramas que menos aportan**: las que casi no reducen el error pero agregan complejidad. `ccp_alpha` regula cuán exigente es ese corte: con 0 no poda nada, y cuanto más grande, más ramas elimina. Se usa en la Parte B del TP2.

### Importancia de las variables

`feature_importances_` indica cuánto contribuyó cada variable a reducir la impureza en todas sus divisiones. Suma 1 entre todas.

En nuestro árbol de regresión:

| Variable | Importancia |
|---|---|
| RyD_Spend | **0.9890** |
| MarketingSpend | 0.0065 |
| Administration | 0.0045 |
| Las tres de State | 0.0000 |

Importancia 0 significa que el árbol **nunca** la usó para dividir. Ojo: si dos variables están muy correlacionadas (como `RyD_Spend` y `MarketingSpend`, con 0.98), el árbol usa una y la otra queda con importancia baja, aunque sea igual de informativa.

### Ventajas y desventajas

**A favor:**
- **Se puede leer y explicar**: es su gran virtud. Se puede mostrar el árbol y seguir el camino de una predicción.
- No necesita escalado (por eso los notebooks de la unidad 4 no estandarizan).
- Acepta variables numéricas y categóricas.
- Resiste los outliers: parte por umbrales, así que un valor altísimo no "tira" del modelo como en una regresión lineal.

**En contra:**
- Sobreajusta con facilidad si no se lo limita.
- **Inestable**: cambiar unas pocas filas puede dar un árbol bastante distinto.
- Sus fronteras son escalonadas (cortes paralelos a los ejes), así que aproxima mal una relación diagonal.

---

## 23. Métricas de regresión

Las tres responden lo mismo —cuánto se equivoca el modelo— pero tratan distinto a los errores grandes.

### Ejemplo hecho a mano

Cinco empresas, con su ganancia real y la estimada:

| Empresa | Real | Estimado | Error | \|Error\| | Error² |
|---|---|---|---|---|---|
| 1 | 100.000 | 99.000 | −1.000 | 1.000 | 1.000.000 |
| 2 | 120.000 | 120.500 | +500 | 500 | 250.000 |
| 3 | 90.000 | 89.800 | −200 | 200 | 40.000 |
| 4 | 150.000 | 149.700 | −300 | 300 | 90.000 |
| 5 | 80.000 | 90.000 | +10.000 | 10.000 | **100.000.000** |

**MAE** = promedio de la columna |Error| = (1.000 + 500 + 200 + 300 + 10.000) / 5 = **2.400**

**MSE** = promedio de la columna Error² = 101.380.000 / 5 = **20.276.000**

**RMSE** = √20.276.000 = **4.503**

Miren lo que pasó: cuatro empresas tienen errores chiquitos y una se desvió 10.000. El MAE dice 2.400; el RMSE dice 4.503, casi el doble. **El caso feo pesa mucho más en el RMSE.**

### Por qué se elevan al cuadrado los errores

Dos motivos:

1. **Para que no se cancelen.** Si una predicción se pasa por 1.000 y otra se queda corta por 1.000, el promedio simple daría 0 y parecería perfecto. El cuadrado (o el valor absoluto) lo evita.
2. **Para castigar más los errores grandes.** Equivocarse por 10.000 cuenta 100 veces más que por 1.000, no 10 veces. En muchos problemas eso tiene sentido: un error enorme hace más daño que varios chicos.

### Las tres, comparadas

| | Fórmula | Unidades | Sensible a errores grandes | ¿Se interpreta directo? |
|---|---|---|---|---|
| **MAE** | promedio de \|real − predicho\| | Las de la variable | Poco | **Sí**: "en promedio me equivoco por tanto" |
| **MSE** | promedio de (real − predicho)² | **Al cuadrado** | Mucho | No: está en pesos² |
| **RMSE** | √MSE | Las de la variable | Mucho | Sí |

**Por qué suele usarse RMSE y no MSE:** está en las mismas unidades que la variable, así que se puede comparar con su escala. El MSE queda en "pesos al cuadrado", que no significa nada intuitivo.

**Por qué igual se calcula el MSE:** es la función que los modelos minimizan internamente, porque es derivable (el valor absoluto del MAE no lo es en el cero).

### La brecha entre MAE y RMSE es información

El RMSE **siempre** es mayor o igual que el MAE. Lo informativo es cuánto:

| Relación | Qué significa |
|---|---|
| RMSE ≈ MAE | Todos los errores son parecidos |
| RMSE bastante mayor | Hay unos pocos errores grandes entre muchos chicos |

En nuestro TP2 (80/20): MAE 997 y RMSE 3.585, más del triple. Eso anticipa lo que después muestra la Figura 5: casi todos los puntos pegados a la diagonal, y un puñado lejos.

### Cómo saber si un error es "bueno"

**Un MAE de 997 no significa nada por sí solo.** Hay que compararlo con la escala de la variable:

- Ganancia media: 118.675 → el error es **menos del 1%**. Muy bueno.
- Si la variable promediara 2.000, ese mismo error sería del 50%. Malísimo.

Siempre hay que relativizar. Una forma rápida es comparar el RMSE con el **desvío estándar** de la variable objetivo: si el RMSE es mucho menor, el modelo explica bastante; si se le parece, el modelo no es mejor que predecir siempre el promedio.

### R²

El **coeficiente de determinación** (filmina 67 de la unidad 4) mide qué proporción de la variabilidad de la variable objetivo logra explicar el modelo.

| Valor | Significa |
|---|---|
| **1** | Perfecto: explica toda la variabilidad |
| **0** | No es mejor que predecir siempre el promedio |
| **Negativo** | Peor que predecir el promedio |

Es cómodo porque no depende de las unidades, así que permite comparar problemas distintos. El enunciado del TP2 no lo pide, pero es el `scoring` por defecto de sklearn para regresores.

---

## 24. Preguntas típicas de parcial

**¿Por qué hay que estandarizar antes de PCA o de un clustering?**
Porque trabajan con varianzas y distancias. Sin estandarizar, la variable con los números más grandes domina el resultado solo por su unidad de medida.

**¿Qué es la varianza explicada de una componente?**
La proporción de la variabilidad total de los datos que captura esa componente.

**¿Cuántas componentes conservo?**
Las que indique el criterio elegido: varianza acumulada del 75-80%, eigenvalue > 1 (Kaiser) o el codo. Alcanza con **uno** de los tres, justificándolo.

**En el método del codo, ¿la componente del quiebre se incluye?**
**Sí.** Se conservan todas hasta la del quiebre, incluyéndola. Vale igual para elegir el k de K-means. Es lo que nos corrigió el profesor en el TP1 (sección 25).

**¿Qué es un eigenvalue en PCA?**
La varianza de la componente. Con datos estandarizados, uno mayor que 1 significa que esa componente aporta más que una variable original.

**¿Las componentes están correlacionadas entre sí?**
No, son independientes por construcción: cada una aporta información nueva.

**Si PCA separa bien las clases, ¿es porque las conoce?**
No. PCA es no supervisado: no ve la variable objetivo. Si los grupos aparecen separados, es porque las diferencias entre ellos son la principal fuente de variabilidad de los datos.

**¿Por qué no eliminar siempre los outliers?**
Porque pueden ser valores reales y propios de una clase. Al eliminarlos se pierde justamente lo que distingue a esa clase.

**¿Por qué imputar por grupo y no por columna?**
Porque el promedio general sesga las muestras hacia el centro y borra las diferencias entre grupos.

**¿Qué es la paradoja de Simpson?**
Que una tendencia observada sobre los datos mezclados se invierte al mirar dentro de cada grupo. Nos pasó con la profundidad del culmen.

**¿En qué se diferencian PCA e Isomap?**
PCA es lineal y preserva la varianza; Isomap es no lineal y preserva las distancias geodésicas, medidas saltando de vecino en vecino por un grafo.

**¿Qué pasa si `n_neighbors` es muy grande en Isomap?**
Se crean atajos entre puntos que no son vecinos reales, las distancias geodésicas se parecen a las euclídeas y el resultado se aproxima al de PCA.

**¿Qué es una matriz dispersa y por qué se usa para el grafo de vecinos?**
Una matriz donde casi todo son ceros, de la que se guardan solo los valores distintos de cero. El grafo de vecinos es disperso porque cada punto se conecta con unos pocos vecinos y el resto de la matriz son ceros.

**¿Qué significa el aviso de que el grafo tiene más de una componente conexa?**
Que con esa cantidad de vecinos quedaron grupos sin camino entre sí. sklearn completa el grafo, pero esas distancias son artificiales: conviene aumentar `n_neighbors`.

**¿Por qué dos corridas de t-SNE dan gráficos distintos?**
Porque parte de una configuración aleatoria. La calidad (el KL) se mantiene, pero la disposición cambia. Se fija `random_state` para que sea reproducible.

**¿Se puede decir que dos clusters de un gráfico t-SNE están muy lejos entre sí?**
No. t-SNE expande las zonas densas y comprime las dispersas, así que ni las distancias entre grupos ni sus tamaños relativos son interpretables.

**¿Por qué no se puede usar el KL para elegir la perplejidad?**
Porque la perplejidad define la tabla de vecindades contra la que se mide el error: al cambiarla se compara contra otra referencia. Solo sirve para comparar iteraciones o componentes. De todos modos el KL no se vio en clase y no conviene usarlo para justificar decisiones en una entrega (sección 25).

**¿Qué método elegirías para cada cosa?**
PCA si se necesita interpretar qué variables pesan y cuánta información se conserva; t-SNE si solo se quiere ver si existen grupos; Isomap si la estructura es curva y se quiere conservar la noción de distancia.

**¿Qué es la inercia y por qué no alcanza para elegir k?**
La suma de distancias al cuadrado de cada punto a su centroide. Siempre baja al aumentar k, así que solo sirve buscar el codo, que a veces no es nítido.

**¿Cómo se interpreta el coeficiente de Silhouette?**
Va de −1 a 1. Cerca de 1, clusters compactos y separados; cerca de 0, superpuestos; negativo, puntos mal asignados. Se elige el k con el promedio más alto.

**¿Qué compara el estadístico GAP?**
La inercia de los datos reales contra la de datos uniformes al azar generados en el mismo rango. Se elige el k donde la diferencia (en logaritmos) es mayor.

**¿Por qué K-means dio 6 clusters si hay 3 especies?**
Porque es no supervisado y solo ve distancias. Al incluir `Sexo` como característica, los grupos naturales pasan a ser especie × sexo.

**¿El número de cluster significa algo?**
No. Las etiquetas 0, 1, 2... son arbitrarias; hay que cruzarlas con lo que se conoce de los datos (tabla cruzada).

**¿Cómo se elige el número de clusters en un dendrograma?**
Trazando una línea horizontal donde haya un tramo vertical largo sin uniones y contando las ramas que corta.

**¿Qué significa la altura de una unión en el dendrograma?**
La distancia entre los dos grupos que se unen. Baja: eran parecidos. Alta: eran distintos.

**¿Qué es el enlace `ward`?**
El criterio que une, en cada paso, los dos clusters cuya fusión menos aumenta la dispersión interna. Da grupos compactos, parecidos a los de K-means.

**¿Qué hacer si Silhouette y GAP indican números distintos?**
Mirar cuánta diferencia hay entre los valores, qué dicen los otros indicadores (dendrograma) y qué grupos se forman con cada k. A nosotros GAP dio 7 por 0.007 y elegimos 6.

**¿Qué problema trae el desbalance de clases?**
Que la exactitud engaña: predecir siempre la clase mayoritaria ya da un valor alto. Hay que estratificar, mirar las métricas por clase y reportarlas en macro.

**¿Por qué hay que separar en entrenamiento y test?**
Para medir si el modelo generaliza. Evaluarlo con los datos con los que aprendió premia al que memoriza.

**¿Qué es la fuga de datos y cómo se evita?**
Que información del test se filtre al entrenamiento. Se evita separando primero y ajustando todas las transformaciones (escalado, codificación, imputación, discretización) **solo con el entrenamiento**.

**¿Cómo se comprueba que el escalado se hizo bien?**
El entrenamiento queda con media 0 y desvío 1; el test, no exactamente. Si el test también diera 0 y 1, el escalador se habría ajustado con todos los datos.

**¿Para qué sirve la validación cruzada si ya tengo un conjunto de test?**
Para elegir hiperparámetros sin tocar el test. Si se eligiera mirando el test, dejaría de ser una medición imparcial.

**¿Qué hace `GridSearchCV`?**
Prueba todas las combinaciones de una grilla, evalúa cada una por validación cruzada dentro del entrenamiento y devuelve la mejor.

**¿Cómo decide un árbol por dónde cortar?**
Prueba todos los cortes posibles y elige el que deja los dos grupos más homogéneos: por error cuadrático en regresión, por Gini o entropía en clasificación.

**¿De dónde salen los umbrales con decimales, como 82.150,5?**
Son los puntos medios entre valores consecutivos de la variable. El árbol genera un candidato por cada par de valores consecutivos y los evalúa todos.

**¿Por qué se dice que el árbol es un algoritmo voraz?**
Porque en cada nodo elige el mejor corte del momento, sin mirar si uno peor permitiría mejores cortes más abajo. No garantiza el árbol óptimo global.

**¿Qué diferencia hay entre Gini y entropía?**
Las dos miden impureza de un nodo: 0 si es puro. Gini es el criterio por defecto de sklearn; en la práctica dan resultados muy parecidos.

**¿Cómo se evita que un árbol sobreajuste?**
Limitando su crecimiento (`max_depth`, `min_samples_leaf`, `min_samples_split`) o podándolo después (`ccp_alpha`).

**¿Por qué los árboles no necesitan estandarización?**
Porque parten por umbrales sobre cada variable por separado; cambiar la escala no altera el orden de los valores ni, por lo tanto, los cortes.

**¿Qué significa que MAE y RMSE estén muy separados?**
Que hay unas pocas predicciones con error grande entre muchas muy buenas: el RMSE castiga más esos casos.

**¿Por qué el MSE no se interpreta directamente?**
Porque queda en unidades al cuadrado. Para leerlo en las unidades de la variable se usa el RMSE.

**¿Media o mediana para imputar?**
La mediana, si hay asimetría o valores extremos, porque no se deja arrastrar por ellos.

---

## 25. Lo que nos corrigió el profesor en el TP1

El TP1 se aprobó con **7**. La devolución llegó el 16/10/2026, con el trabajo ya entregado, así que el notebook no se modificó: lo que sigue es para no repetirlo en el TP2 ni en el parcial.

**Lo que estuvo bien:** el análisis descriptivo, el tratamiento de nulos y duplicados, Isomap entero, y la justificación de cuántas componentes conservar en PCA.

**1. El codo incluye el punto del quiebre.** Nuestra frase "el quiebre se produce en la tercera componente, por lo que conservamos las anteriores: 2 componentes" está mal: por ese criterio eran **3**. Lo mismo en K-means, donde escribimos que la inercia bajaba de forma pronunciada hasta k=6 y el quiebre en realidad se lee en 4 o 5. Es la corrección más importante, porque es una regla y no una interpretación. Aclaró que, cuando el codo no alcanza, está bien seguir con Silhouette y GAP: el k=6 final no fue objetado.

**2. Los barridos de parámetros se declaran completos.** Para t-SNE pidió una figura variando la perplejidad con las iteraciones fijas y otra variando las iteraciones con la perplejidad fija, para terminar eligiendo **la mejor combinación** de los dos. Eso fue lo que hicimos (perplejidad con `max_iter=1000`, iteraciones con `perplexity=15`), pero los valores fijos quedaron escritos solo en el título de las figuras y el texto decía "con el resto de los parámetros fijos". Conclusión práctica: **el valor fijo va en el texto**, y el cierre del barrido nombra la combinación elegida, no un parámetro a la vez.

**3. No se decide con métricas que no se vieron en clase.** Usamos el KL para elegir la cantidad de componentes de t-SNE sin explicar qué es ni qué significa que baje. Y explicarlo tampoco era salida, porque el enunciado prohíbe incluir teoría en el informe. Lo que corresponde es decidir con los criterios de clase y con lo que se ve en los gráficos.

**4. Hay que mostrar cómo quedan los grupos, en todos los métodos.** En K-means cruzamos los clusters con especie y sexo (Tablas 20 y 21) y además los graficamos en 3D con los centroides (Figura 17). En el jerárquico quedaron las tablas (24 y 25) pero no el gráfico, y lo marcó como que no mostramos los grupos. Cada método de clustering cierra con la composición de sus clusters **y** su visualización.

**Lo transversal:** tres de las cuatro observaciones son de redacción, no de código. Los cálculos estaban bien; lo que falló fue cómo se leyó el gráfico (1), qué se dejó sin aclarar (2) y qué se usó para justificar (3).

---

## 26. Temas que todavía no vimos

Se van a agregar a este archivo, con la misma explicación para principiantes, cuando los trabajemos:

**Del TP2, todavía sin usar:**
- Poda de árboles con `ccp_alpha` (Parte B, actividad 4)
- Métricas de clasificación: precisión, exhaustividad, exactitud, macro contra micro (actividad 4)
- Matriz de confusión (actividad 4)
- Desbalance de clases (actividad 3)
- Naive Bayes y discretización por cuantiles (actividad 5)
- k-NN: distancias, elección de k (actividad 6)

**Vistos en clase pero fuera de los TPs:** MDS, UMAP, DBSCAN, HDBSCAN y SVM (unidad 5).

---

## 27. Resumen de una carilla

Para repasar el día antes. Todo lo de acá está explicado en detalle en las secciones anteriores.

### Los dos mundos

| | **No supervisado** (unidades 2 y 3) | **Supervisado** (unidades 4 y 5) |
|---|---|---|
| ¿Ve la respuesta? | No | Sí, al entrenar |
| Objetivo | Encontrar estructura | Predecir |
| Métodos | PCA, Isomap, t-SNE, K-means, jerárquico | Árboles, Naive Bayes, k-NN, SVM |
| Evaluación | Silhouette, GAP | MAE/RMSE (regresión), exactitud/precisión/exhaustividad (clasificación) |

### Preparación de datos: el orden

```
1. Explorar  →  2. Limpiar  →  3. SEPARAR train/test  →  4. Codificar  →  5. Escalar
                                      ↑
                 todo lo que aprende algo de los datos va DESPUÉS de acá,
                 y se ajusta (fit) SOLO con el entrenamiento
```

- **Estandarizar** (z-score): `(valor − media) / desvío` → media 0, desvío 1.
- **Siempre antes** de PCA, Isomap, t-SNE, K-means, jerárquico y k-NN. **No hace falta** en árboles.
- **Categóricas:** con orden → `OrdinalEncoder`; sin orden → one-hot.
- **Comprobación de que no hubo fuga:** el test **no** debe quedar con media 0 y desvío 1.

### Reducción de la dimensionalidad

| | PCA | Isomap | t-SNE |
|---|---|---|---|
| Tipo | Lineal | No lineal | No lineal |
| Preserva | Varianza (global) | Distancias geodésicas | Vecindades (local) |
| Parámetro clave | — | `n_neighbors` | `perplexity` (5-50) |
| Ejes interpretables | **Sí** (cargas) | No | No |
| Determinista | Sí | Sí | **No** |
| Cuidado | — | k chico: grafo desconectado; k grande: atajos | No leer tamaños ni distancias entre grupos |

**Cuántas componentes en PCA, los 3 criterios:**
1. **Varianza acumulada:** llegar al 75-80%.
2. **Kaiser:** eigenvalues > 1.
3. **Codo:** donde la curva se aplana. ⚠️ **La componente del quiebre se incluye.**

Alcanza con justificar **uno**.

### Clustering

| | K-means | Jerárquico |
|---|---|---|
| Cómo funciona | Asignar al centroide más cercano y recalcular, hasta estabilizar | Unir de a dos los grupos más parecidos |
| Hay que fijar k antes | Sí | No: se corta el dendrograma después |
| Determinista | No (`random_state`) | Sí |
| Salida extra | Centroides, inercia | Dendrograma |

**Cómo elegir k:**
- **Codo:** inercia contra k; se corta donde se aplana. ⚠️ **El k del quiebre se incluye.**
- **Silhouette:** `s = (b − a) / max(a, b)`, de −1 a 1. Más alto, mejor. Se busca el máximo.
- **GAP:** compara la dispersión real contra la de datos aleatorios uniformes **en el rango de cada variable**. Se busca el máximo.

**Enlaces del jerárquico:** `single` (mínimo), `complete` (máximo), `average` (promedio), `ward` (el que menos aumenta la dispersión interna; el más usado).

### Supervisado: el esquema

```
   ENTRENAMIENTO (80%)                    TEST (20%)
         │                                     │
   GridSearchCV con cv=5:                  guardado
   prueba combinaciones,                   bajo llave
   elige por validación cruzada                │
         │                                     │
   mejor modelo ───────── predict ────────────→│
                                          métricas finales
```

- **`random_state=63`** en todo (lo pide el enunciado del TP2).
- **`stratify=y`** en clasificación, para conservar la proporción de clases.
- Los puntajes `neg_...` de sklearn son negativos: se dan vuelta para leerlos.

### Árboles de decisión

- **Cómo elige un corte:** prueba todos y se queda con el que deja los hijos más homogéneos, pesando por cuántas muestras tiene cada uno.
- **Impureza:** Gini `= 1 − Σpᵢ²` o entropía (clasificación); error cuadrático (regresión). Nodo puro = 0.
- **Hoja:** responde el promedio (regresión) o la clase mayoritaria (clasificación).
- **Hiperparámetros:** `max_depth`, `min_samples_leaf` (mínimo que queda en la hoja), `min_samples_split` (mínimo para intentar partir), `ccp_alpha` (poda posterior).
- **Sobreajuste:** error casi 0 en entrenamiento y alto en test. **Subajuste:** alto en los dos.
- **Ventaja principal:** se puede leer y explicar. **Desventaja:** inestable y sobreajusta fácil.

### Métricas

**Regresión:**

| | Qué es | Unidades |
|---|---|---|
| MAE | Promedio de \|error\| | Las de la variable |
| MSE | Promedio de error² | Al cuadrado |
| RMSE | √MSE | Las de la variable |

RMSE mucho mayor que MAE → hay pocos errores grandes entre muchos chicos. Siempre comparar el error con la escala de la variable.

**Clasificación** (se ven en la Parte B del TP2): exactitud, precisión, exhaustividad, matriz de confusión; con clases desbalanceadas se reportan en **macro** (promedio por clase, sin pesar por tamaño).

### Las trampas que ya nos costaron puntos

1. **El codo incluye la componente o el k del quiebre.**
2. **No usar métricas que no se vieron en clase** para justificar decisiones (nos pasó con el KL).
3. **Mostrar siempre cómo quedan los grupos**, con tabla y gráfico, en todos los métodos de clustering.
4. **Escribir los valores fijos** cuando se barre un parámetro ("perplejidad 15, variando las iteraciones").
5. **Citar solo números que se ven** en una figura, tabla o salida del notebook.
