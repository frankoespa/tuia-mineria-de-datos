# Repaso para el parcial — Minería de Datos

Explicaciones para principiantes de **todos** los conceptos que vamos aplicando en los trabajos prácticos, con los ejemplos de nuestros propios datos (pingüinos).

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
18. [Preguntas típicas de parcial](#18-preguntas-típicas-de-parcial)
19. [Temas que todavía no vimos](#19-temas-que-todavía-no-vimos)

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

**Más de dos categorías:** codificarlas como 1, 2, 3 es peligroso, porque el algoritmo interpretaría que hay un orden y que 3 está "más lejos" de 1 que de 2. Para eso existe el *one-hot encoding*, que crea una columna por categoría con 0 o 1. (No lo necesitamos en el TP1.)

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
| **Codo (scree)** | Cortar donde la curva se quiebra y se aplana | Quiebre en PC3 | **2** |

**Qué es un eigenvalue (valor propio):** la varianza de esa componente medida en "unidades de variable original". Como estandarizamos, cada variable original aporta exactamente 1. Por eso Kaiser dice: *si una componente vale menos de 1, aporta menos que una sola variable original y no justifica conservarla*.

Diferencia entre las dos columnas que devuelve sklearn:
- `explained_variance_` = **eigenvalues** (2.87, 1.38, …) → es el que usa Kaiser.
- `explained_variance_ratio_` = la misma información como **porcentaje** (57.2%, 27.5%, …) → es el de la varianza acumulada.

> ⚠️ El notebook de clase U2 aplica Kaiser (celda 98) como "varianza explicada ≥ 0.1", que **no** es el criterio de Kaiser enunciado en la celda 91. Con nuestros datos los dos dan 2 componentes, pero en el parcial conviene usar **eigenvalue > 1**.

### Los gráficos de PCA

**a) Varianza explicada y acumulada.** Las barras muestran cuánto aporta cada componente; la línea roja, cuánto se lleva acumulado. Se busca el primer punto de la línea que cruza el 75-80%.

**b) Gráfico del codo (scree).** La varianza de cada componente dibujada como curva que baja. Se busca **el quiebre**: el punto donde deja de caer bruscamente y se aplana. Se conservan las componentes **anteriores** al codo. Se llama así porque la curva parece un brazo doblado.

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

Es la medida de **cuánto se diferencian las dos tablas**, y lo que el algoritmo minimiza. KL = 0 significa que el mapa reproduce las vecindades a la perfección; cuanto más chico, mejor copiado.

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

**`max_iter` (iteraciones).** Cuántas veces mueve los puntos. Lo habitual es del orden de 1000. Si se corta antes de converger, aparecen formas raras y "pellizcadas", con grupos poco definidos. Nuestros números: KL 0.549 (300), 0.383 (500), 0.361 (1000), 0.353 (2000). A partir de 1000 deja de mejorar.

⚠️ En `scikit-learn` moderno el parámetro se llama **`max_iter`**; antes era `n_iter`. El mínimo admitido es 250, pero no debe usarse: ahí la optimización termina durante la fase inicial (*early exaggeration*) y el KL informado es un valor centinela gigantesco, no un error real.

**`n_components`.** Dimensiones de salida, 2 o 3 para visualizar.

**`random_state` (semilla).** Como arranca al azar, cada corrida da un dibujo distinto. Fijar la semilla hace el resultado reproducible. Con otra semilla **cambia el dibujo, no la calidad** (lo probamos con tres semillas y el KL casi no varió, aunque ese apartado no quedó en el TP). Por eso se recomienda correrlo varias veces y comparar.

### Qué NO hay que leer en un gráfico de t-SNE

Esto suele ser pregunta de parcial:

1. **Los ejes no significan nada.** No hay cargas ni interpretación posible, a diferencia de PCA.
2. **Los tamaños de los grupos no son reales.** El algoritmo expande las zonas densas y comprime las dispersas, así que todos los grupos terminan pareciendo del mismo tamaño.
3. **Las distancias entre grupos no son confiables.** Que dos grupos aparezcan lejos no significa que sean muy distintos.

Lo que sí se puede afirmar: **que los grupos existen y están diferenciados**.

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

El problema es que **siempre baja al aumentar k** (con un cluster por punto sería 0), así que no se puede elegir "el k con menor inercia". Se grafica contra k y se busca el **codo**: el punto donde deja de bajar rápido.

En nuestros datos: 1710 (k=1), 907, 593, 403, 301, **233 (k=6)**, 217, 203... Baja de a poco hasta 6 y después se aplana. No hay un codo nítido, y por eso hacen falta índices más precisos.

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

## 18. Preguntas típicas de parcial

**¿Por qué hay que estandarizar antes de PCA o de un clustering?**
Porque trabajan con varianzas y distancias. Sin estandarizar, la variable con los números más grandes domina el resultado solo por su unidad de medida.

**¿Qué es la varianza explicada de una componente?**
La proporción de la variabilidad total de los datos que captura esa componente.

**¿Cuántas componentes conservo?**
Las que indique el criterio elegido: varianza acumulada del 75-80%, eigenvalue > 1 (Kaiser) o el quiebre del codo. Alcanza con **uno** de los tres, justificándolo.

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
Porque la perplejidad define la tabla de vecindades contra la que se mide el error: al cambiarla se compara contra otra referencia. Solo sirve para comparar iteraciones o componentes.

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

**¿Media o mediana para imputar?**
La mediana, si hay asimetría o valores extremos, porque no se deja arrastrar por ellos.

---

## 19. Temas que todavía no vimos

Se van a agregar a este archivo, con la misma explicación para principiantes, cuando los trabajemos:

- Vistos en clase pero fuera del TP1: MDS, UMAP, DBSCAN, HDBSCAN, división en entrenamiento y prueba (TP2)
