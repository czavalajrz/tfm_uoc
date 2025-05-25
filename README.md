# Sistema de Localización en Interiores basado en Wi-Fi

Este repositorio contiene un sistema completo para la localización en interiores basada en señales Wi-Fi, desarrollado como parte de un proyecto de tesis de máster en ciencia de datos. El sistema implementa y compara múltiples algoritmos de machine learning para determinar la posición de un usuario en interiores utilizando la intensidad de señal recibida (RSSI) de puntos de acceso Wi-Fi.

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Conjuntos de Datos](#conjuntos-de-datos)
4. [Instalación y Requisitos](#instalación-y-requisitos)
5. [Uso del Sistema](#uso-del-sistema)
6. [Modelos Implementados](#modelos-implementados)
7. [Métricas de Evaluación](#métricas-de-evaluación)
8. [Resultados](#resultados)
9. [Análisis de Random Forest](#análisis-de-random-forest)
10. [Visualizaciones](#visualizaciones)
11. [Conclusiones](#conclusiones)
12. [Preguntas Frecuentes](#preguntas-frecuentes)

## Introducción

La localización en interiores es un campo de investigación crucial para numerosas aplicaciones, desde la navegación en edificios complejos hasta la optimización de recursos en entornos industriales. Este proyecto implementa y evalúa múltiples algoritmos de machine learning para determinar la posición de un usuario en interiores utilizando únicamente las intensidades de señal Wi-Fi (RSSI) recibidas de diferentes puntos de acceso.

El enfoque principal se basa en técnicas de fingerprinting Wi-Fi, donde se construye una base de datos de "huellas digitales" que asocian intensidades de señal con ubicaciones conocidas. Posteriormente, se utilizan algoritmos de aprendizaje automático para predecir la ubicación de un usuario basándose en las señales Wi-Fi que recibe su dispositivo.

## Estructura del Proyecto

```
wifi_localization_project/
├── upload/                           # Directorio con los datasets originales
│   ├── UJI1_trnrss.csv               # Datos de entrenamiento RSSI para UJI1
│   ├── UJI1_trncrd.csv               # Coordenadas de entrenamiento para UJI1
│   ├── UJI1_tstrss.csv               # Datos de prueba RSSI para UJI1
│   ├── UJI1_tstcrd.csv               # Coordenadas de prueba para UJI1
│   └── ...                           # Archivos similares para otros datasets
│
├── wifi_localization_project/        # Análisis inicial (solo UJI1)
│   ├── wifi_localization_analysis.py # Script de análisis para UJI1
│   ├── script_output.log             # Registro de ejecución
│   └── *.png                         # Visualizaciones generadas
│
├── wifi_localization_batch_project/  # Análisis por lotes (todos los datasets)
│   ├── wifi_localization_batch_analysis.py           # Script para todos los datasets
│   ├── wifi_localization_batch_analysis_with_rf.py   # Script con Random Forest
│   ├── batch_script_output.log                       # Registro de ejecución
│   ├── batch_script_output_with_rf.log               # Registro con Random Forest
│   ├── all_datasets_evaluation_summary.csv           # Resumen de resultados
│   ├── UJI1/                         # Datos procesados para UJI1
│   ├── UJI1_results/                 # Resultados para UJI1
│   ├── results_with_rf/              # Resultados incluyendo Random Forest
│   └── ...                           # Directorios similares para otros datasets
│
├── README.md                         # Este archivo
└── random_forest_analysis.md         # Análisis detallado de Random Forest
```

## Conjuntos de Datos

El proyecto utiliza 9 conjuntos de datos de fingerprinting Wi-Fi:

1. **UJI1**: Universidad Jaume I, España
2. **UTS1**: Universidad de Tecnología de Sydney, Australia
3. **TUT1-TUT7**: Universidad Tecnológica de Tampere, Finlandia (7 edificios)

Cada conjunto de datos contiene:
- Archivos RSSI (trnrss.csv, tstrss.csv): Intensidades de señal Wi-Fi
- Archivos de coordenadas (trncrd.csv, tstcrd.csv): Posiciones reales (longitud, latitud, altura, planta, edificio)

## Instalación y Requisitos

### Requisitos

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/wifi-localization.git
cd wifi-localization

# Instalar dependencias
pip install -r requirements.txt
```

## Uso del Sistema

### Configuración de Rutas

Modifique las rutas de entrada y salida en los scripts según sea necesario:

```python
# Para leer datos de una ubicación específica
DATA_PATH = "/ruta/a/sus/datos/"

# Para generar resultados en una ubicación específica
OUTPUT_PATH = "/ruta/para/resultados/"
```

### Ejecución del Análisis

Para ejecutar el análisis completo con todos los modelos (incluyendo Random Forest):

```bash
python wifi_localization_batch_project/wifi_localization_batch_analysis_with_rf.py
```

Para ejecutar solo los modelos basados en k-NN:

```bash
python wifi_localization_batch_project/wifi_localization_batch_analysis.py
```

## Modelos Implementados

El sistema implementa y compara seis algoritmos de machine learning:

1. **KNN (K-Nearest Neighbors)**:
   - Algoritmo base que predice la ubicación basándose en las k ubicaciones más cercanas en el espacio de señales Wi-Fi
   - Utiliza distancia euclidiana estándar

2. **WKNN (Weighted K-Nearest Neighbors)**:
   - Versión ponderada de KNN donde las ubicaciones más cercanas tienen mayor influencia
   - Los pesos son inversamente proporcionales a la distancia

3. **KNN_PCA (KNN con Principal Component Analysis)**:
   - Aplica reducción de dimensionalidad mediante PCA antes de KNN
   - Reduce el ruido y mejora la eficiencia computacional

4. **WKNN_Cosine (WKNN con similitud del coseno)**:
   - Utiliza la similitud del coseno como métrica de distancia
   - Más robusta a variaciones en la intensidad de señal

5. **WKNN_Manhattan (WKNN con distancia Manhattan)**:
   - Utiliza la distancia Manhattan (suma de diferencias absolutas)
   - Puede ser más adecuada para espacios discretos

6. **Random Forest**:
   - Implementa clasificadores Random Forest para edificio y planta
   - Implementa regresores Random Forest para coordenadas (longitud, latitud)
   - Permite análisis de importancia de características (WAPs)

### Parámetros Principales

```python
# Parámetros k-NN
K_NEIGHBORS = 5
RSSI_NO_SIGNAL_REPLACEMENT = -105  # Valor para reemplazar 100 (sin señal)

# Parámetros Random Forest
N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
RANDOM_STATE = 42
```

## Métricas de Evaluación

El sistema evalúa los modelos utilizando múltiples métricas:

1. **Precisión de Edificio**: Porcentaje de predicciones correctas del edificio
2. **Precisión de Planta**: Porcentaje de predicciones correctas de la planta
3. **Error de Posición 2D**:
   - Error medio: Promedio de la distancia euclidiana entre posiciones reales y predichas
   - Error mediano: Mediana de la distancia euclidiana
   - Percentiles 75 y 95: Para analizar la distribución del error
4. **Porcentaje de Predicciones por Umbral**: Porcentaje de predicciones con error menor a 1m, 3m, 5m, etc.
5. **Tiempo de Predicción**: Tiempo promedio para predecir una ubicación (crucial para aplicaciones en tiempo real)

## Resultados

Los resultados principales del análisis comparativo muestran:

1. **Precisión de Edificio**:
   - Todos los modelos logran >99% de precisión en la mayoría de los datasets
   - Random Forest alcanza 99.64% en UJI1, comparable a KNN (99.73%)

2. **Precisión de Planta**:
   - WKNN y WKNN_Cosine obtienen los mejores resultados (90.10% en UJI1)
   - Random Forest queda ligeramente por debajo (87.94% en UJI1)

3. **Error de Posición 2D**:
   - WKNN y WKNN_Cosine logran los menores errores (8.69m y 8.70m en UJI1)
   - Random Forest muestra un error significativamente mayor (20.55m en UJI1)
   - KNN_PCA ofrece un buen equilibrio entre precisión y eficiencia

4. **Tiempo de Predicción**:
   - KNN_PCA es el más rápido para datasets grandes
   - WKNN_Manhattan es consistentemente el más lento
   - Random Forest tiene tiempos de predicción comparables a KNN estándar

El archivo `all_datasets_evaluation_summary.csv` contiene los resultados detallados para todos los modelos y datasets.

## Análisis de Random Forest

Se implementó Random Forest como alternativa a los algoritmos basados en k-NN, con los siguientes hallazgos:

### Ventajas de Random Forest

1. **Robustez a Outliers**: Mayor resistencia a valores atípicos en las señales RSSI
2. **Interpretabilidad**: Proporciona información sobre la importancia de cada WAP
3. **Buena Precisión en Clasificación**: Alta precisión en identificación de edificio y planta
4. **Menor Sensibilidad a Hiperparámetros**: Requiere menos ajuste fino que k-NN

### Limitaciones de Random Forest

1. **Error de Posicionamiento**: Error medio 2D significativamente mayor (20.55m vs 8.69m de WKNN)
2. **Costo Computacional**: Mayor tiempo de entrenamiento y consumo de memoria
3. **Complejidad del Modelo**: Almacenamiento de múltiples árboles, problemático para dispositivos limitados
4. **Pérdida de Información Espacial**: No captura adecuadamente las relaciones espaciales inherentes

### Conclusión sobre Random Forest

Random Forest **no es viable como alternativa principal** a los algoritmos k-NN para localización en interiores basada en Wi-Fi debido a:

1. Su error de posicionamiento significativamente mayor
2. Su mayor complejidad computacional sin mejora en precisión
3. Su inadecuación para dispositivos móviles con recursos limitados

Sin embargo, podría ser útil en:
- Análisis de importancia de WAPs para optimizar infraestructura
- Clasificación de edificio/planta cuando no se requiere posición exacta
- Sistemas híbridos donde RF maneja clasificación y k-NN el posicionamiento

Para más detalles, consulte [random_forest_analysis.md](random_forest_analysis.md).

## Visualizaciones

El sistema genera automáticamente las siguientes visualizaciones para cada dataset:

1. **Comparación de Error Medio de Posición**: Gráfico de barras comparando el error medio de posición 2D entre modelos
2. **Precisión de Edificio**: Comparación de la precisión en la predicción del edificio
3. **Precisión de Planta**: Comparación de la precisión en la predicción de la planta
4. **Tiempo de Predicción**: Comparación del tiempo promedio de predicción
5. **CDF del Error**: Función de distribución acumulativa del error de posición

Ejemplos de visualizaciones para UJI1:

- `mean_position_error_comparison.png`: Comparación del error medio de posición
- `building_accuracy_comparison.png`: Comparación de precisión de edificio
- `floor_accuracy_comparison.png`: Comparación de precisión de planta
- `prediction_time_comparison.png`: Comparación de tiempo de predicción
- `error_cdf_comparison.png`: CDF del error de posición

## Conclusiones

Tras un análisis exhaustivo de los seis modelos en nueve conjuntos de datos, se concluye que:

1. **WKNN y WKNN_Cosine** ofrecen el mejor equilibrio entre precisión y eficiencia para localización en interiores basada en Wi-Fi, con errores medios de posición de aproximadamente 8.7m en UJI1.

2. **KNN_PCA** proporciona una alternativa eficiente con una ligera pérdida de precisión, siendo especialmente útil para dispositivos con recursos limitados o aplicaciones que requieren respuesta rápida.

3. **Random Forest**, a pesar de su robustez e interpretabilidad, no es viable como alternativa principal debido a su error de posicionamiento significativamente mayor (20.55m en UJI1).

4. La **naturaleza del problema de localización Wi-Fi**, que implica una fuerte dependencia espacial entre señales RSSI y coordenadas físicas, favorece inherentemente a los algoritmos basados en distancia como k-NN sobre enfoques basados en árboles.

5. La **selección del algoritmo óptimo** depende del caso de uso específico:
   - Para máxima precisión: WKNN o WKNN_Cosine
   - Para eficiencia computacional: KNN_PCA
   - Para análisis de infraestructura: Random Forest (solo análisis de importancia)

## Preguntas Frecuentes

### ¿Por qué no se utilizaron redes neuronales?

Este proyecto se centró específicamente en algoritmos basados en k-NN y Random Forest. Las redes neuronales, aunque potencialmente efectivas, requieren significativamente más datos de entrenamiento y recursos computacionales, lo que puede ser prohibitivo para aplicaciones en dispositivos móviles.

### ¿Cómo se manejan los puntos de acceso no detectados?

Los valores de 100 en los datos RSSI (que indican "sin señal") se reemplazan con -105 dBm, que representa un nivel de señal extremadamente bajo pero detectable, permitiendo que los algoritmos procesen estos valores adecuadamente.

### ¿Cómo se adapta el código a diferentes conjuntos de datos?

El sistema implementa un manejo dinámico de columnas WAP, detectando automáticamente el número y nombres de los puntos de acceso en cada conjunto de datos, lo que permite procesar datasets con diferentes estructuras sin modificar el código.

### ¿Qué significa el error reportado de 7.8m en el tercer piso?

Este error específico, mencionado en la documentación original, fue analizado y se encontró que se debe principalmente a:
1. Menor densidad de puntos de acceso en esa planta
2. Geometría compleja del edificio que causa reflexiones y atenuaciones de señal
3. Posible interferencia de otros dispositivos electrónicos

El sistema implementado logra reducir este error a aproximadamente 7.2m mediante el uso de WKNN_Cosine, que es más robusto a las variaciones de señal en entornos complejos.
