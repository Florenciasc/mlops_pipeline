# Pipeline MLOps – Predicción de Riesgo Crediticio (FinTech)

## Descripción General

Este proyecto desarrolla un pipeline completo de Machine Learning bajo principios de **MLOps**, orientado a la predicción del comportamiento de pago de clientes en un entorno FinTech.

El objetivo es anticipar si un cliente pagará su crédito en tiempo y forma, permitiendo mejorar la gestión de riesgo crediticio y la toma de decisiones basada en datos.

Variable objetivo:

- `pago_atiempo`
  - 1 → Pago en tiempo y forma
  - 0 → Incumplimiento / Mora

El desarrollo contempla:

- Análisis exploratorio profundo
- Ingeniería de características
- Evaluación comparativa de modelos
- Despliegue como API
- Monitoreo estadístico (Data Drift)
- Containerización con Docker

---

# 🏗 Arquitectura del Sistema
Datos históricos
↓
EDA
↓
Feature Engineering
↓
Entrenamiento y validación
↓
Selección del modelo
↓
API (FastAPI)
↓
Monitoreo (Drift)
↓
Visualización (Streamlit)


---

# Análisis Exploratorio de Datos (EDA)

Se realizó:

- Revisión de estructura y tipos de datos
- Unificación y tratamiento de valores nulos
- Análisis univariable (distribuciones, medidas estadísticas)
- Análisis bivariable respecto al target
- Matriz de correlación y análisis multivariable

### Hallazgo relevante

Se identificó que la variable `puntaje` presentaba una correlación muy alta (~0.92) con la variable objetivo.

Esto sugiere posible **fuga de información (data leakage)**, por lo que fue excluida del modelo productivo para evitar sobreestimación artificial del desempeño.

---

# Ingeniería de Características

Se implementó un pipeline utilizando `ColumnTransformer` que incluye:

- Imputación de variables numéricas
- Codificación de variables categóricas
- Separación consistente de train/test
- Transformaciones reproducibles entre entrenamiento e inferencia

El diseño garantiza consistencia entre el flujo de entrenamiento y el flujo de predicción en producción.

---

# Modelamiento y Evaluación

Se evaluaron los siguientes modelos:

- Regresión Logística (balanceada)
- Random Forest
- XGBoost

Métricas utilizadas:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

En el contexto de riesgo crediticio, se priorizó:

- Capacidad de detección (Recall)
- Equilibrio general (F1 Score)
- Estabilidad del modelo

## Modelo Seleccionado

**RandomForest sin la variable `puntaje`**

Justificación:

- Buen equilibrio entre precision y recall
- Desempeño estable
- Sin evidencia de fuga de información
- Mayor robustez ante ruido y outliers

---

# Despliegue del Modelo (FastAPI)

El modelo se expone como servicio REST utilizando FastAPI.

### Ejecutar localmente

uvicorn src.model_deploy:app --reload

Endpoints disponibles:

/health

/predict

/docs

Soporta predicción por lote (batch scoring) y retorna probabilidades junto con clasificación binaria.

Monitoreo y Detección de Data Drift

Se implementó un módulo de monitoreo para detectar cambios en la distribución de los datos que puedan afectar el desempeño del modelo.

## Métricas aplicadas:

Variables numéricas:

Kolmogorov-Smirnov (KS)

Population Stability Index (PSI)

Jensen-Shannon Divergence

Variables categóricas:

Test Chi-cuadrado

Ejecución: 
python src/model_monitoring.py

Genera reportes de drift y permite seguimiento histórico.

## Visualización (Streamlit)

Se desarrolló una aplicación interactiva que permite:

Visualizar métricas de drift

Analizar variables con desviaciones significativas

Comparar distribuciones históricas vs actuales

Identificar tendencias en el tiempo

Ejecución:
Visualización (Streamlit)

Se desarrolló una aplicación interactiva que permite:

Visualizar métricas de drift

Analizar variables con desviaciones significativas

Comparar distribuciones históricas vs actuales

Identificar tendencias en el tiempo

Ejecución:
streamlit run src/app.py

## Containerización

El proyecto incluye Docker para asegurar portabilidad y consistencia entre entornos.

Construcción:
docker build -t mlops-api .

Ejecución:
docker run -p 8000:8000 mlops-api

## Buenas Prácticas Implementadas

Separación de responsabilidades (train vs deploy)

Detección de data leakage

Transformaciones reproducibles

Monitoreo estadístico continuo

Preparación para integración CI/CD

Containerización para entornos productivos

## Stack Tecnológico

Python 3.11

Pandas / NumPy

Scikit-learn

XGBoost

FastAPI

Streamlit

Docker

**Autora**

Florencia Sosa Comisso
Proyecto Integrador – Pipeline MLOps aplicado a Riesgo Crediticio
